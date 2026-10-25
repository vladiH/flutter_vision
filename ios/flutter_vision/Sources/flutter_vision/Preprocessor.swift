// Copyright 2022-2026 The flutter_vision authors. All rights reserved.
// Use of this source code is governed by the MIT license found in the LICENSE file.

import CoreGraphics
import Foundation
import ImageIO

/// Letterbox transform: inputPixel = imagePixel * scale + pad.
struct Letterbox {
  let imageWidth: Int
  let imageHeight: Int
  let scale: Double
  let padX: Double
  let padY: Double

  /// Fits width x height inside the model input, centred, keeping the aspect ratio.
  static func fit(width: Int, height: Int, inputWidth: Int, inputHeight: Int) -> Letterbox {
    let scale = min(Double(inputWidth) / Double(width), Double(inputHeight) / Double(height))
    return Letterbox(
      imageWidth: width, imageHeight: height, scale: scale,
      padX: (Double(inputWidth) - Double(width) * scale) / 2,
      padY: (Double(inputHeight) - Double(height) * scale) / 2)
  }
}

/// Builds the model input from an encoded image or a camera frame.
/// Owned by one model and used only from that model's queue.
final class Preprocessor {
  let inputWidth: Int
  let inputHeight: Int
  let channelsFirst: Bool
  let int8: Bool

  private let context: CGContext
  private(set) var floatInput: [Float]
  private(set) var int8Input: [Int8]
  private var rgba: [UInt8] = []

  init(inputWidth: Int, inputHeight: Int, channelsFirst: Bool, int8: Bool) throws {
    self.inputWidth = inputWidth
    self.inputHeight = inputHeight
    self.channelsFirst = channelsFirst
    self.int8 = int8
    let count = inputWidth * inputHeight * 3
    floatInput = int8 ? [] : [Float](repeating: 0, count: count)
    int8Input = int8 ? [Int8](repeating: 0, count: count) : []
    guard
      let context = CGContext(
        data: nil, width: inputWidth, height: inputHeight, bitsPerComponent: 8,
        bytesPerRow: inputWidth * 4, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
          | CGBitmapInfo.byteOrder32Big.rawValue)
    else {
      throw VisionError.loadFailed("Cannot allocate the input image buffer.")
    }
    context.interpolationQuality = .high
    self.context = context
  }

  /// Decodes an image, applies the EXIF orientation and fills the input.
  func fromEncoded(_ data: Data) throws -> Letterbox {
    guard let source = CGImageSourceCreateWithData(data as CFData, nil),
      let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
      let rawWidth = properties[kCGImagePropertyPixelWidth] as? Int,
      let rawHeight = properties[kCGImagePropertyPixelHeight] as? Int,
      rawWidth > 0, rawHeight > 0
    else {
      throw VisionError.invalidInput("Cannot decode the image (unknown or corrupted format).")
    }
    let orientation = (properties[kCGImagePropertyOrientation] as? UInt32) ?? 1
    let transposed = (5...8).contains(orientation)
    let width = transposed ? rawHeight : rawWidth
    let height = transposed ? rawWidth : rawHeight
    let fit = Letterbox.fit(
      width: width, height: height, inputWidth: inputWidth, inputHeight: inputHeight)

    // Decode directly at (at least) the size we need, already upright.
    let target = Int((Double(max(width, height)) * min(1, fit.scale * 2)).rounded(.up))
    let options: [CFString: Any] = [
      kCGImageSourceCreateThumbnailFromImageAlways: true,
      kCGImageSourceCreateThumbnailWithTransform: true,
      kCGImageSourceShouldCacheImmediately: true,
      kCGImageSourceThumbnailMaxPixelSize: max(target, 1),
    ]
    guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
    else {
      throw VisionError.invalidInput("Cannot decode the image.")
    }
    draw(image, rotation: 0, fit: fit)
    return fit
  }

  /// Converts a camera frame, rotates it upright and fills the input.
  func fromFrame(_ frame: FrameMessage) throws -> Letterbox {
    let width = Int(frame.width)
    let height = Int(frame.height)
    guard width > 0, height > 0, !frame.planes.isEmpty else {
      throw VisionError.invalidInput("Empty camera frame.")
    }
    let rotation = ((Int(frame.rotationDegrees) % 360) + 360) % 360
    let planes = frame.planes
    let image: CGImage
    switch frame.format {
    case .bgra8888:
      // Fast path: wrap the BGRA bytes without converting them.
      image = try makeImage(
        bytes: planes[0].bytes.data, width: width, height: height,
        bytesPerRow: Int(planes[0].bytesPerRow),
        bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
          | CGBitmapInfo.byteOrder32Little.rawValue)
    case .rgba8888:
      image = try makeImage(
        bytes: planes[0].bytes.data, width: width, height: height,
        bytesPerRow: Int(planes[0].bytesPerRow),
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
          | CGBitmapInfo.byteOrder32Big.rawValue)
    case .nv21:
      try ImageConverter.biPlanar(
        y: [UInt8](planes[0].bytes.data),
        uv: [UInt8](planes[0].bytes.data.dropFirst(Int(planes[0].bytesPerRow) * height)),
        width: width, height: height, yRowStride: Int(planes[0].bytesPerRow),
        uvRowStride: Int(planes[0].bytesPerRow), vFirst: true, out: &rgba)
      image = try rgbaImage(width: width, height: height)
    case .yuv420:
      if planes.count == 2 {
        try ImageConverter.biPlanar(
          y: [UInt8](planes[0].bytes.data), uv: [UInt8](planes[1].bytes.data),
          width: width, height: height, yRowStride: Int(planes[0].bytesPerRow),
          uvRowStride: Int(planes[1].bytesPerRow), vFirst: false, out: &rgba)
      } else if planes.count == 3 {
        try ImageConverter.planar(
          y: [UInt8](planes[0].bytes.data), u: [UInt8](planes[1].bytes.data),
          v: [UInt8](planes[2].bytes.data), width: width, height: height,
          yRowStride: Int(planes[0].bytesPerRow), uvRowStride: Int(planes[1].bytesPerRow),
          uvPixelStride: max(1, Int(planes[1].bytesPerPixel)), out: &rgba)
      } else {
        throw VisionError.invalidInput("YUV420 frames need 2 or 3 planes, got \(planes.count).")
      }
      image = try rgbaImage(width: width, height: height)
    }
    let uprightWidth = rotation % 180 == 0 ? width : height
    let uprightHeight = rotation % 180 == 0 ? height : width
    let fit = Letterbox.fit(
      width: uprightWidth, height: uprightHeight, inputWidth: inputWidth,
      inputHeight: inputHeight)
    draw(image, rotation: rotation, fit: fit)
    return fit
  }

  private func makeImage(
    bytes: Data, width: Int, height: Int, bytesPerRow: Int, bitmapInfo: UInt32
  ) throws -> CGImage {
    let stride = max(bytesPerRow, width * 4)
    guard bytes.count >= (height - 1) * stride + width * 4,
      let provider = CGDataProvider(data: bytes as CFData),
      let image = CGImage(
        width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
        bytesPerRow: stride, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo), provider: provider, decode: nil,
        shouldInterpolate: true, intent: .defaultIntent)
    else {
      throw VisionError.invalidInput("Frame buffer smaller than \(width)x\(height).")
    }
    return image
  }

  private func rgbaImage(width: Int, height: Int) throws -> CGImage {
    try makeImage(
      bytes: Data(rgba), width: width, height: height, bytesPerRow: width * 4,
      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
  }

  /// Draws [image] rotated clockwise by [rotation] into the letterboxed input
  /// and converts the pixels into the tensor.
  private func draw(_ image: CGImage, rotation: Int, fit: Letterbox) {
    let gray: CGFloat = 114.0 / 255.0
    context.setFillColor(red: gray, green: gray, blue: gray, alpha: 1)
    context.fill(CGRect(x: 0, y: 0, width: inputWidth, height: inputHeight))

    let width = Double(fit.imageWidth) * fit.scale
    let height = Double(fit.imageHeight) * fit.scale
    // CoreGraphics has its origin at the bottom left.
    let centerX = fit.padX + width / 2
    let centerY = Double(inputHeight) - (fit.padY + height / 2)
    let drawWidth = rotation % 180 == 0 ? width : height
    let drawHeight = rotation % 180 == 0 ? height : width
    context.saveGState()
    context.translateBy(x: centerX, y: centerY)
    context.rotate(by: -Double(rotation) * .pi / 180)
    context.draw(
      image,
      in: CGRect(x: -drawWidth / 2, y: -drawHeight / 2, width: drawWidth, height: drawHeight))
    context.restoreGState()

    guard let data = context.data else { return }
    let pixels = data.bindMemory(to: UInt8.self, capacity: inputWidth * inputHeight * 4)
    Preprocessor.fillTensor(
      pixels: pixels, count: inputWidth * inputHeight, channelsFirst: channelsFirst,
      floats: &floatInput, bytes: &int8Input, int8: int8)
  }

  /// Writes RGBX pixels as RGB: float32 in 0...1, or int8 as pixel - 128.
  static func fillTensor(
    pixels: UnsafePointer<UInt8>, count: Int, channelsFirst: Bool,
    floats: inout [Float], bytes: inout [Int8], int8: Bool
  ) {
    let inv: Float = 1 / 255
    for i in 0..<count {
      let r = pixels[i * 4]
      let g = pixels[i * 4 + 1]
      let b = pixels[i * 4 + 2]
      let ir = channelsFirst ? i : 3 * i
      let ig = channelsFirst ? count + i : 3 * i + 1
      let ib = channelsFirst ? 2 * count + i : 3 * i + 2
      if int8 {
        bytes[ir] = Int8(truncatingIfNeeded: Int(r) - 128)
        bytes[ig] = Int8(truncatingIfNeeded: Int(g) - 128)
        bytes[ib] = Int8(truncatingIfNeeded: Int(b) - 128)
      } else {
        floats[ir] = Float(r) * inv
        floats[ig] = Float(g) * inv
        floats[ib] = Float(b) * inv
      }
    }
  }
}
