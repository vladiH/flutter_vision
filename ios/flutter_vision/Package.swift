// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

// LiteRT 2.3.0 prebuilt binaries, from the official release:
// https://github.com/google-ai-edge/LiteRT/releases/tag/v2.3.0
let liteRtRelease = "https://github.com/google-ai-edge/LiteRT/releases/download/v2.3.0"

let package = Package(
    name: "flutter_vision",
    platforms: [
        .iOS("15.0")
    ],
    products: [
        .library(name: "flutter-vision", targets: ["flutter_vision"])
    ],
    dependencies: [
        .package(name: "FlutterFramework", path: "../FlutterFramework")
    ],
    targets: [
        .target(
            name: "flutter_vision",
            dependencies: [
                .product(name: "FlutterFramework", package: "FlutterFramework"),
                "FlutterVisionLiteRT",
                // Loaded at runtime by LiteRT for the GPU (Metal) accelerator.
                "LiteRtMetalAccelerator",
            ],
            resources: [
                .process("PrivacyInfo.xcprivacy")
            ]
        ),
        // LiteRT's Swift API (Apache 2.0), vendored from
        // google-ai-edge/LiteRT v2.3.1 litert/swift/Sources/LiteRT so apps do
        // not need to clone the whole LiteRT repository.
        .target(
            name: "FlutterVisionLiteRT",
            dependencies: ["CLiteRT"],
            exclude: ["LICENSE"]
        ),
        .binaryTarget(
            name: "CLiteRT",
            url: "\(liteRtRelease)/CLiteRT.xcframework.zip",
            checksum: "b36a1a4f1b3eeb04546b4cc2986902cd704439b37c34597ab0f804aed08f5a00"
        ),
        .binaryTarget(
            name: "LiteRtMetalAccelerator",
            url: "\(liteRtRelease)/LiteRtMetalAccelerator.xcframework.zip",
            checksum: "e564e6678959098a35c1dbcffc60041afac25a845abcee998a53f958e342aeec"
        ),
    ]
)
