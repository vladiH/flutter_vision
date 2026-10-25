// Copyright 2022-2026 The flutter_vision authors. All rights reserved.
// Use of this source code is governed by the MIT license found in the LICENSE file.

import Flutter
import UIKit

/// Entry point. Keeps one `ModelSession` per loaded model. All methods run on
/// the main thread; the work itself runs on each session's queue.
public class FlutterVisionPlugin: NSObject, FlutterPlugin, FlutterVisionHostApi {
  private let registrar: FlutterPluginRegistrar
  private var sessions: [Int64: ModelSession] = [:]
  private var nextHandle: Int64 = 1

  init(registrar: FlutterPluginRegistrar) {
    self.registrar = registrar
  }

  public static func register(with registrar: FlutterPluginRegistrar) {
    let instance = FlutterVisionPlugin(registrar: registrar)
    FlutterVisionHostApiSetup.setUp(binaryMessenger: registrar.messenger(), api: instance)
    registrar.publish(instance)
  }

  public func detachFromEngine(for registrar: FlutterPluginRegistrar) {
    FlutterVisionHostApiSetup.setUp(binaryMessenger: registrar.messenger(), api: nil)
    for session in sessions.values {
      session.close {}
    }
    sessions.removeAll()
  }

  func loadModel(
    request: LoadModelRequest, completion: @escaping (Result<ModelInfoMessage, Error>) -> Void
  ) {
    let path: String
    if request.isAsset {
      let key = registrar.lookupKey(forAsset: request.modelPath)
      guard let assetPath = Bundle.main.path(forResource: key, ofType: nil) else {
        completion(
          .failure(
            ModelSession.pigeonError(
              .notFound(
                "Asset '\(request.modelPath)' not found. Check that it is listed under flutter/assets in pubspec.yaml."
              ))))
        return
      }
      path = assetPath
    } else {
      path = request.modelPath
    }

    let handle = nextHandle
    nextHandle += 1
    let session = ModelSession(handle: handle)
    sessions[handle] = session
    session.load(path: path, request: request) { [weak self] result in
      if case .failure = result {
        self?.sessions.removeValue(forKey: handle)
        session.close {}
      }
      completion(result)
    }
  }

  func runOnImage(
    handle: Int64, encodedImage: FlutterStandardTypedData,
    completion: @escaping (Result<InferenceResultMessage, Error>) -> Void
  ) {
    guard let session = sessions[handle] else {
      completion(.failure(ModelSession.pigeonError(.disposed)))
      return
    }
    session.runImage(encodedImage.data, completion: completion)
  }

  func runOnFrame(
    handle: Int64, frame: FrameMessage,
    completion: @escaping (Result<InferenceResultMessage, Error>) -> Void
  ) {
    guard let session = sessions[handle] else {
      completion(.failure(ModelSession.pigeonError(.disposed)))
      return
    }
    session.runFrame(frame, completion: completion)
  }

  func disposeModel(handle: Int64, completion: @escaping (Result<Void, Error>) -> Void) {
    guard let session = sessions.removeValue(forKey: handle) else {
      completion(.success(()))
      return
    }
    session.close { completion(.success(())) }
  }
}
