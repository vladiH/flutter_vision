# R8 rules added to apps that use flutter_vision.

# LiteRT's optional AiPackModelProvider references Play AI delivery, which
# flutter_vision excludes (it is not used and its WorkManager dependency
# crashes minified release builds).
-dontwarn com.google.android.play.**
-dontwarn com.google.ai.edge.litert.deployment.**
