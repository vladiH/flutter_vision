import '../exceptions.dart';
import '../types.dart';

/// Size of the model input and whether it is channels-first (NCHW).
class InputLayout {
  const InputLayout({
    required this.width,
    required this.height,
    required this.channelsFirst,
  });

  /// Reads `[1, H, W, 3]` (NHWC) or `[1, 3, H, W]` (NCHW).
  factory InputLayout.fromShape(List<int> shape) {
    final dims = shape.length == 3 ? [1, ...shape] : shape;
    if (dims.length != 4) {
      throw FlutterVisionException(
        FlutterVisionException.unsupportedModel,
        'Expected an image input of shape [1, H, W, 3] or [1, 3, H, W], '
        'got $shape.',
      );
    }
    if (dims[3] == 3) {
      return InputLayout(width: dims[2], height: dims[1], channelsFirst: false);
    }
    if (dims[1] == 3) {
      return InputLayout(width: dims[3], height: dims[2], channelsFirst: true);
    }
    throw FlutterVisionException(
      FlutterVisionException.unsupportedModel,
      'The model input must have 3 color channels, got shape $shape.',
    );
  }

  final int width;
  final int height;
  final bool channelsFirst;
}

/// How to read the outputs of a YOLO model.
class ModelLayout {
  const ModelLayout({
    required this.format,
    required this.task,
    required this.numClasses,
    required this.numMasks,
    required this.predictionsIndex,
    required this.channelsFirst,
    required this.numCandidates,
    required this.numChannels,
    this.protosIndex,
    this.protosChannelsFirst = false,
    this.protoHeight = 0,
    this.protoWidth = 0,
  });

  final YoloFormat format;
  final YoloTask task;

  /// Number of classes encoded in the output. For [YoloFormat.endToEnd] the
  /// output does not tell, so it is the number of labels.
  final int numClasses;

  /// Mask coefficients per candidate (32 for Ultralytics models, 0 when
  /// detecting).
  final int numMasks;

  /// Index of the predictions tensor among the model outputs.
  final int predictionsIndex;

  /// True for `[1, channels, candidates]`, false for `[1, candidates, channels]`.
  final bool channelsFirst;
  final int numCandidates;
  final int numChannels;

  /// Index of the mask prototypes tensor (segmentation only).
  final int? protosIndex;

  /// True for `[1, masks, H, W]`, false for `[1, H, W, masks]`.
  final bool protosChannelsFirst;
  final int protoHeight;
  final int protoWidth;

  /// Offset of the first class score inside a candidate.
  int get classOffset => format == YoloFormat.anchorBased ? 5 : 4;

  /// Offset of the first mask coefficient inside a candidate.
  int get maskOffset => switch (format) {
        YoloFormat.anchorBased => 5 + numClasses,
        YoloFormat.anchorFree => 4 + numClasses,
        YoloFormat.endToEnd => 6,
      };

  /// Works out the layout from the output shapes and the number of labels.
  ///
  /// [labelCount] can be 0 when no labels were given; the layout is then
  /// guessed from the shapes alone.
  static ModelLayout resolve({
    required List<List<int>> outputShapes,
    required int labelCount,
    YoloFormat? format,
    YoloTask? task,
  }) {
    final protoIndices = <int>[];
    final predictionIndices = <int>[];
    for (var i = 0; i < outputShapes.length; i++) {
      final shape = _withoutBatch(outputShapes[i]);
      if (shape.length == 3) {
        protoIndices.add(i);
      } else if (shape.length == 2) {
        predictionIndices.add(i);
      }
    }
    if (predictionIndices.length != 1 || protoIndices.length > 1) {
      throw FlutterVisionException(
        FlutterVisionException.unsupportedModel,
        'Not a YOLO model: expected one predictions output (plus one mask '
        'prototypes output for segmentation), got outputs $outputShapes.',
      );
    }

    final predictionsIndex = predictionIndices.single;
    final pred = _withoutBatch(outputShapes[predictionsIndex]);
    final d1 = pred[0];
    final d2 = pred[1];

    var numMasks = 0;
    int? protosIndex;
    var protosChannelsFirst = false;
    var protoHeight = 0;
    var protoWidth = 0;
    if (protoIndices.isNotEmpty) {
      protosIndex = protoIndices.single;
      final p = _withoutBatch(outputShapes[protosIndex]);
      if (p[2] <= p[0] && p[2] <= p[1]) {
        numMasks = p[2];
        protoHeight = p[0];
        protoWidth = p[1];
      } else {
        protosChannelsFirst = true;
        numMasks = p[0];
        protoHeight = p[1];
        protoWidth = p[2];
      }
    }
    final modelTask = protosIndex == null ? YoloTask.detect : YoloTask.segment;
    if (task != null && task != modelTask) {
      throw FlutterVisionException(
        FlutterVisionException.unsupportedModel,
        task == YoloTask.segment
            ? 'This is a detection model; it cannot do segmentation. '
                'Export a "-seg" model or load it with YoloTask.detect.'
            : 'This is a segmentation model. Load it with YoloTask.segment '
                'or omit the task.',
      );
    }

    ModelLayout build(YoloFormat f, bool channelsFirst, int classes) {
      return ModelLayout(
        format: f,
        task: modelTask,
        numClasses: classes,
        numMasks: numMasks,
        predictionsIndex: predictionsIndex,
        channelsFirst: channelsFirst,
        numCandidates: channelsFirst ? d2 : d1,
        numChannels: channelsFirst ? d1 : d2,
        protosIndex: protosIndex,
        protosChannelsFirst: protosChannelsFirst,
        protoHeight: protoHeight,
        protoWidth: protoWidth,
      );
    }

    // Layouts that agree with the number of labels, most common first.
    final nc = labelCount;
    final nm = numMasks;
    final matches = <ModelLayout>[
      if (nc > 0 && d1 == 4 + nc + nm) build(YoloFormat.anchorFree, true, nc),
      if (d2 == 6 + nm && d1 <= 1000)
        build(YoloFormat.endToEnd, false, nc),
      if (nc > 0 && d2 == 5 + nc + nm) build(YoloFormat.anchorBased, false, nc),
      if (nc > 0 && d2 == 4 + nc + nm) build(YoloFormat.anchorFree, false, nc),
      if (nc > 0 && d1 == 5 + nc + nm) build(YoloFormat.anchorBased, true, nc),
    ];
    for (final layout in matches) {
      if (format == null || layout.format == format) return layout;
    }

    // Nothing agrees with the labels: guess from the shape alone.
    final ModelLayout guess;
    final requested = format;
    if (requested == YoloFormat.endToEnd ||
        (requested == null && d2 == 6 + nm && d1 <= 1000)) {
      final channelsFirst = d1 == 6 + nm && d2 != 6 + nm;
      guess = build(YoloFormat.endToEnd, channelsFirst, nc);
    } else if (requested == YoloFormat.anchorFree ||
        (requested == null && d1 < d2)) {
      final channelsFirst = d1 < d2;
      final channels = channelsFirst ? d1 : d2;
      guess = build(YoloFormat.anchorFree, channelsFirst, channels - 4 - nm);
    } else {
      final channelsFirst = d1 < d2;
      final channels = channelsFirst ? d1 : d2;
      guess = build(YoloFormat.anchorBased, channelsFirst, channels - 5 - nm);
    }

    if (guess.format != YoloFormat.endToEnd) {
      if (guess.numClasses <= 0) {
        throw FlutterVisionException(
          FlutterVisionException.unsupportedModel,
          'Cannot read the model output ${outputShapes[predictionsIndex]} '
          'as ${guess.format.name}.',
        );
      }
      if (nc > 0 && guess.numClasses != nc) {
        throw FlutterVisionException(
          FlutterVisionException.unsupportedModel,
          'The model predicts ${guess.numClasses} classes but the labels '
          'have $nc entries. Check that the labels file belongs to this model '
          '(one label per line).',
        );
      }
    }
    return guess;
  }

  static List<int> _withoutBatch(List<int> shape) {
    if (shape.isNotEmpty && shape.first == 1 && shape.length > 2) {
      return shape.sublist(1);
    }
    return shape;
  }
}
