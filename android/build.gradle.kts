group = "com.vladih.computer_vision.flutter_vision"
version = "3.0.0"

buildscript {
    val kotlinVersion = "2.4.0"
    repositories {
        google()
        mavenCentral()
    }

    dependencies {
        classpath("com.android.tools.build:gradle:9.1.0")
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:$kotlinVersion")
    }
}

allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

plugins {
    id("com.android.library")
}

android {
    namespace = "com.vladih.computer_vision.flutter_vision"

    compileSdk = 36

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    sourceSets {
        getByName("main") {
            java.srcDirs("src/main/kotlin")
        }
        getByName("test") {
            java.srcDirs("src/test/kotlin")
        }
    }

    defaultConfig {
        minSdk = 24
        consumerProguardFiles("consumer-rules.pro")
    }

    testOptions {
        unitTests {
            all {
                it.useJUnitPlatform()
                it.testLogging {
                    events("passed", "skipped", "failed", "standardOut", "standardError")
                    showStandardStreams = true
                }
            }
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

dependencies {
    // LiteRT runtime + CompiledModel API (CPU / NPU) and the GPU accelerator.
    // Play AI packs (model download) are not used; they bring WorkManager and
    // Room, which crash at startup in release builds minified by R8.
    implementation("com.google.ai.edge.litert:litert:2.3.0") {
        exclude(group = "com.google.android.play", module = "ai-delivery")
    }
    implementation("com.google.ai.edge.litert:litert-gpu:2.3.0") {
        exclude(group = "com.google.android.play", module = "ai-delivery")
    }
    implementation("androidx.exifinterface:exifinterface:1.4.2")

    testImplementation("org.jetbrains.kotlin:kotlin-test")
}
