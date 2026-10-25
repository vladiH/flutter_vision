// Copyright 2022-2026 The flutter_vision authors. All rights reserved.
// Use of this source code is governed by the MIT license found in the LICENSE file.

import FlutterVisionLiteRT
import Foundation

#if os(iOS)
  import Flutter
#endif

/// One loaded model. Loading, inference and release all run on the model's
/// own serial queue, so a model is never released while it is running.
final class ModelSession {
  let handle: Int64
  private let queue: DispatchQueue

  /// Only read and written on the main thread.
  private var isClosed = false

  private var environment: Environment?
  private var model: CompiledModel?
  private var inputBuffers: [TensorBuffer] = []
  private var outputBuffers: [TensorBuffer] = []
  private var outputTypes: [TensorType] = []
  private var preprocessor: Preprocessor?

  init(handle: Int64) {
    self.handle = handle
    queue = DispatchQueue(label: "flutter_vision.model.\(handle)", qos: .userInitiated)
  }

  func load(
    path: String, request: LoadModelRequest,
    completion: @escaping (Result<ModelInfoMessage, Error>) -> Void
  ) {
    submit(completion, fallback: { .loadFailed($0) }) {
      try self.loadOnQueue(path: path, request: request)
    }
  }

  func runImage(_ data: Data, completion: @escaping (Result<InferenceResultMessage, Error>) -> Void)
  {
    submit(completion, fallback: { .inferenceFailed($0) }) {
      try self.infer { try $0.fromEncoded(data) }
    }
  }

  func runFrame(
    _ frame: FrameMessage, completion: @escaping (Result<InferenceResultMessage, Error>) -> Void
  ) {
    submit(completion, fallback: { .inferenceFailed($0) }) {
      try self.infer { try $0.fromFrame(frame) }
    }
  }

  /// Releases the model after the calls already queued.
  func close(completion: @escaping () -> Void) {
    if isClosed {
      completion()
      return
    }
    isClosed = true
    queue.async {
      self.release()
      DispatchQueue.main.async { completion() }
    }
  }

  private func submit<T>(
    _ completion: @escaping (Result<T, Error>) -> Void,
    fallback: @escaping (String) -> VisionError,
    work: @escaping () throws -> T
  ) {
    if isClosed {
      completion(.failure(ModelSession.pigeonError(VisionError.disposed)))
      return
    }
    queue.async {
      let result: Result<T, Error>
      do {
        result = .success(try work())
      } catch let error as VisionError {
        result = .failure(ModelSession.pigeonError(error))
      } catch {
        result = .failure(ModelSession.pigeonError(fallback("\(error)")))
      }
      DispatchQueue.main.async { completion(result) }
    }
  }

  static func pigeonError(_ error: VisionError) -> PigeonError {
    PigeonError(code: error.code, message: error.message, details: nil)
  }

  private func loadOnQueue(path: String, request: LoadModelRequest) throws -> ModelInfoMessage {
    guard FileManager.default.fileExists(atPath: path) else {
      throw VisionError.notFound("Model file not found: \(path)")
    }
    let environment = try Environment()
    self.environment = environment

    var preference = request.accelerators
    if preference.isEmpty { preference = [.cpu] }
    var lastError: Error?
    for index in preference.indices {
      var accelerators: Set<Accelerator> = [.cpu]
      for accelerator in preference[index...] {
        switch accelerator {
        case .cpu: accelerators.insert(.cpu)
        case .gpu: accelerators.insert(.gpu)
        case .npu: accelerators.insert(.npu)
        }
      }
      do {
        let options = try Options()
        try options.setHardwareAccelerators(accelerators)
        let cpu = try CpuOptions()
        try cpu.setNumThreads(max(1, Int(request.numThreads)))
        try options.addConcreteOptions(cpu)
        let compiled = try CompiledModel(filePath: path, environment: environment, options: options)
        model = compiled
        let info = try prepare(compiled, accelerator: preference[index])
        try warmUp()
        return info
      } catch let error as VisionError {
        // Unsupported model: no point trying other accelerators.
        releaseModel()
        throw error
      } catch {
        lastError = error
        releaseModel()
      }
    }
    throw VisionError.loadFailed(
      "LiteRT could not load the model: \(lastError.map { "\($0)" } ?? "unknown error")")
  }

  private func prepare(_ compiled: CompiledModel, accelerator: AcceleratorMessage) throws
    -> ModelInfoMessage
  {
    let inputCount = try compiled.inputCount()
    guard inputCount == 1 else {
      throw VisionError.unsupportedModel(
        "Expected one image input, the model has \(inputCount).")
    }
    let inputType = try compiled.inputTensorType(inputIndex: 0)
    let inputElement = elementType(inputType)
    guard inputElement != .unsupported else {
      throw VisionError.unsupportedModel("The model input must be float32 or int8.")
    }
    let shape = inputType.layout.dimensions
    let channelsFirst = shape.count == 4 && shape[1] == 3 && shape[3] != 3
    guard shape.count == 4, channelsFirst || shape[3] == 3 else {
      throw VisionError.unsupportedModel(
        "Expected an image input [1, H, W, 3] or [1, 3, H, W], got \(shape).")
    }
    preprocessor = try Preprocessor(
      inputWidth: channelsFirst ? shape[3] : shape[2],
      inputHeight: channelsFirst ? shape[2] : shape[1],
      channelsFirst: channelsFirst,
      int8: inputElement == .int8)

    outputTypes = try (0..<compiled.outputCount()).map {
      try compiled.outputTensorType(outputIndex: $0)
    }
    for type in outputTypes where elementType(type) == .unsupported {
      throw VisionError.unsupportedModel(
        "The model outputs must be float32 or int8. Export the model with float outputs.")
    }
    inputBuffers = try compiled.createInputBuffers()
    outputBuffers = try compiled.createOutputBuffers()

    return ModelInfoMessage(
      handle: handle,
      input: TensorInfoMessage(shape: shape.map { Int64($0) }, elementType: inputElement),
      outputs: outputTypes.map {
        TensorInfoMessage(
          shape: $0.layout.dimensions.map { Int64($0) }, elementType: elementType($0))
      },
      accelerator: accelerator)
  }

  /// One run with a blank input: surfaces accelerator failures at load time.
  private func warmUp() throws {
    try writeInput()
    try model!.run(inputs: inputBuffers, outputs: outputBuffers)
  }

  private func writeInput() throws {
    guard let preprocessor, let buffer = inputBuffers.first else {
      throw VisionError.disposed
    }
    if preprocessor.int8 {
      try buffer.write(preprocessor.int8Input)
    } else {
      try buffer.write(preprocessor.floatInput)
    }
  }

  private func infer(_ prepare: (Preprocessor) throws -> Letterbox) throws
    -> InferenceResultMessage
  {
    guard let model, let preprocessor else { throw VisionError.disposed }
    let start = DispatchTime.now().uptimeNanoseconds
    let letterbox = try prepare(preprocessor)
    try writeInput()
    let prepared = DispatchTime.now().uptimeNanoseconds
    try model.run(inputs: inputBuffers, outputs: outputBuffers)
    var outputs: [FlutterStandardTypedData] = []
    for (buffer, type) in zip(outputBuffers, outputTypes) {
      outputs.append(FlutterStandardTypedData(bytes: try readOutput(buffer, type: type)))
    }
    let done = DispatchTime.now().uptimeNanoseconds
    return InferenceResultMessage(
      outputs: outputs,
      imageWidth: Int64(letterbox.imageWidth),
      imageHeight: Int64(letterbox.imageHeight),
      scale: letterbox.scale,
      padX: letterbox.padX,
      padY: letterbox.padY,
      preprocessMicros: Int64((prepared - start) / 1000),
      inferenceMicros: Int64((done - prepared) / 1000))
  }

  /// Output as little-endian float32 bytes; int8 outputs are dequantized.
  private func readOutput(_ buffer: TensorBuffer, type: TensorType) throws -> Data {
    let count = type.layout.elementCount ?? 0
    if type.elementType == .int8, let quant = type.quantization?.perTensor {
      let raw: [Int8] = try buffer.read()
      let zeroPoint = Float(quant.zeroPoint)
      let floats = raw.prefix(count).map { (Float($0) - zeroPoint) * quant.scale }
      return floats.withUnsafeBufferPointer { Data(buffer: $0) }
    }
    let data = try buffer.read()
    return data.prefix(count * MemoryLayout<Float>.size)
  }

  private func elementType(_ type: TensorType) -> ElementTypeMessage {
    switch type.elementType {
    case .float32:
      return .float32
    case .int8:
      return type.quantization?.perTensor != nil ? .int8 : .unsupported
    default:
      return .unsupported
    }
  }

  private func releaseModel() {
    inputBuffers = []
    outputBuffers = []
    outputTypes = []
    model = nil
    preprocessor = nil
  }

  private func release() {
    releaseModel()
    environment = nil
  }
}
