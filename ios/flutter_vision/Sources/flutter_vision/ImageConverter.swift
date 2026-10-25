// Copyright 2022-2026 The flutter_vision authors. All rights reserved.
// Use of this source code is governed by the MIT license found in the LICENSE file.

import Foundation

/// Converts camera frames to RGBX (4 bytes per pixel, R first).
/// Pure Swift so it can be unit tested on any platform.
enum ImageConverter {
  /// Full-range BT.601 (JFIF) YUV to RGB.
  @inline(__always)
  static func yuvToRgb(_ y: Int, _ u: Int, _ v: Int) -> (UInt8, UInt8, UInt8) {
    let cb = u - 128
    let cr = v - 128
    let r = y + ((91881 * cr) >> 16)
    let g = y - ((22554 * cb + 46802 * cr) >> 16)
    let b = y + ((116130 * cb) >> 16)
    return (clamp(r), clamp(g), clamp(b))
  }

  @inline(__always)
  private static func clamp(_ value: Int) -> UInt8 {
    UInt8(truncatingIfNeeded: value < 0 ? 0 : (value > 255 ? 255 : value))
  }

  /// Y plane plus one interleaved chroma plane. [vFirst] is true for NV21
  /// (V, U) and false for NV12 (U, V, the iOS `420YpCbCr8BiPlanar` layout).
  static func biPlanar(
    y: [UInt8], uv: [UInt8], width: Int, height: Int,
    yRowStride: Int, uvRowStride: Int, vFirst: Bool,
    out: inout [UInt8]
  ) throws {
    let lastUv = ((height + 1) / 2 - 1) * uvRowStride + ((width + 1) / 2 - 1) * 2 + 1
    guard y.count >= (height - 1) * yRowStride + width, uv.count > lastUv else {
      throw VisionError.invalidInput("YUV planes are smaller than the frame size.")
    }
    out = [UInt8](repeating: 255, count: width * height * 4)
    for row in 0..<height {
      let yRow = row * yRowStride
      let uvRow = (row >> 1) * uvRowStride
      var o = row * width * 4
      for col in 0..<width {
        let uvIndex = uvRow + (col & ~1)
        let first = Int(uv[uvIndex])
        let second = Int(uv[uvIndex + 1])
        let (r, g, b) = yuvToRgb(
          Int(y[yRow + col]), vFirst ? second : first, vFirst ? first : second)
        out[o] = r
        out[o + 1] = g
        out[o + 2] = b
        o += 4
      }
    }
  }

  /// Three planes with their own strides (Android `YUV_420_888` layout).
  static func planar(
    y: [UInt8], u: [UInt8], v: [UInt8], width: Int, height: Int,
    yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
    out: inout [UInt8]
  ) throws {
    let lastUv = ((height + 1) / 2 - 1) * uvRowStride + ((width + 1) / 2 - 1) * uvPixelStride
    guard y.count >= (height - 1) * yRowStride + width, u.count > lastUv, v.count > lastUv else {
      throw VisionError.invalidInput("YUV planes are smaller than the frame size.")
    }
    out = [UInt8](repeating: 255, count: width * height * 4)
    for row in 0..<height {
      let yRow = row * yRowStride
      let uvRow = (row >> 1) * uvRowStride
      var o = row * width * 4
      for col in 0..<width {
        let uvIndex = uvRow + (col >> 1) * uvPixelStride
        let (r, g, b) = yuvToRgb(Int(y[yRow + col]), Int(u[uvIndex]), Int(v[uvIndex]))
        out[o] = r
        out[o + 1] = g
        out[o + 2] = b
        o += 4
      }
    }
  }

  /// 4 bytes per pixel in BGRA ([bgr] true) or RGBA order, possibly padded rows.
  static func packed32(
    _ data: [UInt8], width: Int, height: Int, rowStride: Int, bgr: Bool,
    out: inout [UInt8]
  ) throws {
    let stride = max(rowStride, width * 4)
    guard data.count >= (height - 1) * stride + width * 4 else {
      throw VisionError.invalidInput("Frame buffer smaller than \(width)x\(height).")
    }
    out = [UInt8](repeating: 255, count: width * height * 4)
    for row in 0..<height {
      var i = row * stride
      var o = row * width * 4
      for _ in 0..<width {
        out[o] = bgr ? data[i + 2] : data[i]
        out[o + 1] = data[i + 1]
        out[o + 2] = bgr ? data[i] : data[i + 2]
        i += 4
        o += 4
      }
    }
  }
}

/// Error with a code understood by the Dart side (see FlutterVisionException).
enum VisionError: Error {
  case notFound(String)
  case unsupportedModel(String)
  case loadFailed(String)
  case invalidInput(String)
  case inferenceFailed(String)
  case disposed

  var code: String {
    switch self {
    case .notFound: return "not_found"
    case .unsupportedModel: return "unsupported_model"
    case .loadFailed: return "load_failed"
    case .invalidInput: return "invalid_input"
    case .inferenceFailed: return "inference_failed"
    case .disposed: return "disposed"
    }
  }

  var message: String {
    switch self {
    case .notFound(let m), .unsupportedModel(let m), .loadFailed(let m),
      .invalidInput(let m), .inferenceFailed(let m):
      return m
    case .disposed:
      return "The model was disposed."
    }
  }
}
