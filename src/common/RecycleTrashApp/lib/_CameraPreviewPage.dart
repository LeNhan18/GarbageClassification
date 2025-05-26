import 'dart:async';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:google_fonts/google_fonts.dart';
import 'dart:typed_data'; // Để dùng Uint8List
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img_lib; // Đặt bí danh

// Hàm main và MyApp giữ nguyên như code của bạn
void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await _requestCameraPermission();
  final cameras = await availableCameras();
  final firstCamera = cameras.isNotEmpty ? cameras.first : null;

  if (firstCamera != null) {
    runApp(MyApp(camera: firstCamera));
  } else {
    print("No camera found!");
    // Cân nhắc hiển thị thông báo cho người dùng ở đây nếu không có camera
  }
}

Future<void> _requestCameraPermission() async {
  try {
    PermissionStatus status = await Permission.camera.request();
    if (status.isDenied) {
      print("Camera permission denied");
      // Yêu cầu lại một lần nữa nếu muốn
      status = await Permission.camera.request();
      if (status.isDenied) {
        print("Camera permission denied again");
        return;
      }
    }
    if (status.isPermanentlyDenied) {
      print("Camera permission permanently denied");
      await openAppSettings(); // Mở cài đặt ứng dụng để người dùng cấp quyền thủ công
    }
  } catch (e) {
    print("Error requesting camera permission: $e");
  }
}

class MyApp extends StatelessWidget {
  final CameraDescription camera;
  const MyApp({super.key, required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Phân Loại Rác AI',
      theme: ThemeData(
        primarySwatch: Colors.green, // Ví dụ đổi màu chủ đạo
        visualDensity: VisualDensity.adaptivePlatformDensity,
        useMaterial3: true,
      ),
      home: CameraPreviewPage(camera: camera),
      debugShowCheckedModeBanner: false,
    );
  }
}
// Kết thúc phần main và MyApp


class CameraPreviewPage extends StatefulWidget {
  final CameraDescription camera;
  const CameraPreviewPage({super.key, required this.camera});

  @override
  _CameraPreviewPageState createState() => _CameraPreviewPageState();
}

class _CameraPreviewPageState extends State<CameraPreviewPage> with WidgetsBindingObserver {
  CameraController? _controller;
  Future<void>? _initializeControllerFuture;
  bool _isFlashOn = false;
  double _zoomLevel = 1.0;
  double _minZoom = 1.0;
  double _maxZoom = 1.0; // Sẽ được cập nhật sau khi camera khởi tạo
  bool _isFrontCamera = false;
  bool _isGridVisible = false;
  List<CameraDescription> cameras = [];
  bool _isCameraInitialized = false; // Đổi tên từ _isInitialized để rõ ràng hơn
  // bool _isCameraDisposed = false; // Có thể không cần biến này nếu quản lý _controller tốt

  // Biến cho TFLite và AI
  Interpreter? _model1Interpreter;
  Interpreter? _model2aInterpreter;
  Interpreter? _model2bInterpreter;

  // CẬP NHẬT TÊN FILE MODEL TFLITE CỦA BẠN Ở ĐÂY
  // static const String model1Path = 'Assets/ML_Models/model_1_btflite'; // Ví dụ
  static const String model2aPath = 'Assets/ML_Models/model2a.tflite'; // Ví dụ
  static const String model2bPath = 'Assets/ML_Models/model2b.tflite'; // Ví dụ


  // CẬP NHẬT ÁNH XẠ NHÃN CHO ĐÚNG VỚI MODEL CỦA BẠN
  // Model 1: Nhị phân
  static const Map<int, String> _model1ClassNames = {
    0: 'Không tái chế',
    1: 'Tái chế'
    // Giả sử: 0 là 'Không tái chế', 1 là 'Tái chế' (cần khớp với lúc huấn luyện)
  };

  // Model 2A: Chi tiết Rác Không Tái Chế
  static const Map<int, String> _model2aClassNames = {
    0: 'Pin',
    1: 'Rác hữu cơ',
    2: 'Vải vụn',
    3: 'Giày dép cũ',
    4: 'Rác thải khác (không tái chế)'
    // Thay thế bằng các lớp thực tế của Model 2A
  };

  // Model 2B: Chi tiết Rác Tái Chế
  static const Map<int, String> _model2bClassNames = {
    0: 'Bìa carton',
    1: 'Thủy tinh',
    2: 'Kim loại',
    3: 'Giấy',
    4: 'Nhựa'
    // Thay thế bằng các lớp thực tế của Model 2B
  };

  String? _predictionResultText;
  bool _isProcessingImage = false;

  // Kích thước ảnh đầu vào cho model
  static const int MODEL_INPUT_WIDTH = 224;
  static const int MODEL_INPUT_HEIGHT = 224;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initializeAndLoadResources();
  }

  Future<void> _initializeAndLoadResources() async {
    await _initializeCamera(widget.camera); // Khởi tạo với camera được truyền vào ban đầu
    await _loadAllModels();
  }


  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    _model1Interpreter?.close();
    _model2aInterpreter?.close();
    _model2bInterpreter?.close();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final CameraController? cameraController = _controller;
    if (cameraController == null || !cameraController.value.isInitialized) {
      return;
    }
    if (state == AppLifecycleState.inactive) {
      cameraController.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initializeCamera(cameraController.description);
    }
  }

  Future<void> _initializeCamera(CameraDescription cameraDescription) async {
    if (_controller != null) {
      await _controller!.dispose(); // Dispose controller cũ nếu có
    }
    _controller = CameraController(
      cameraDescription,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    _controller!.addListener(() {
      if (mounted) setState(() {}); // Cập nhật UI nếu cần (ví dụ: zoom)
    });

    try {
      _initializeControllerFuture = _controller!.initialize();
      await _initializeControllerFuture;
      // Lấy min/max zoom sau khi khởi tạo
      _minZoom = await _controller!.getMinZoomLevel();
      _maxZoom = await _controller!.getMaxZoomLevel();
      _zoomLevel = _minZoom; // Reset zoom về min khi đổi camera

      if (mounted) {
        setState(() {
          _isCameraInitialized = true;
        });
      }
      await _loadAvailableCameras(); // Tải danh sách camera để chuyển đổi
    } catch (e) {
      print("Lỗi khởi tạo camera: $e");
      if (mounted) {
        _showErrorSnackBar('Lỗi khởi tạo camera: $e');
      }
    }
  }

  Future<void> _loadAvailableCameras() async {
    try {
      cameras = await availableCameras();
    } catch (e) {
      print("Lỗi tải danh sách camera: <span class="math-inline">e");
    }
    }
  Future<void> _loadAllModels() async {
  setState(() { _predictionResultText = "Đang tải model AI\.\.\."; });
  try {
  // _model1Interpreter = await Interpreter.fromAsset(model1Path);
  // print\('Model 1 \(</span>{model1Path}) loaded successfully');

  _model2aInterpreter = await Interpreter.fromAsset(model2aPath);
  print('Model 2A (<span class="math-inline">\{model2aPath\}\) loaded successfully'\);
  _model2bInterpreter = await Interpreter.fromAsset(model2bPath);
  print('Model 2B \(</span>{model2bPath}) loaded successfully');
  if (mounted) setState(() { _predictionResultText = "Sẵn sàng phân loại!"; });
  } catch (e) {
  print("Lỗi tải model TFLite: $e");
  if (mounted) {
  setState(() { _predictionResultText = "Lỗi tải model AI!"; });
  _showErrorSnackBar('Lỗi tải model AI: $e');
  }
  }
  }

  ByteBuffer _convertImageToByteBuffer(img_lib.Image imageInput) {
  img_lib.Image resizedImage = img_lib.copyResize(
  imageInput,
  width: MODEL_INPUT_WIDTH,
  height: MODEL_INPUT_HEIGHT,
  interpolation: img_lib.Interpolation.linear,
  );
  var modelInputBuffer = Float32List(1 * MODEL_INPUT_HEIGHT * MODEL_INPUT_WIDTH * 3);
  var bufferIndex = 0;
  for (var y = 0; y < MODEL_INPUT_HEIGHT; y++) {
  for (var x = 0; x < MODEL_INPUT_WIDTH; x++) {
  var pixel = resizedImage.getPixel(x, y);
  modelInputBuffer[bufferIndex++] = pixel.rNormalized; // Giả sử thư viện image đã có rNormalized trả về [0,1]
  modelInputBuffer[bufferIndex++] = pixel.gNormalized; // Nếu không, bạn cần pixel.r / 255.0
  modelInputBuffer[bufferIndex++] = pixel.bNormalized;
  }
  }
  return modelInputBuffer.buffer.asByteBuffer();
  }

  int _getIndexOfMax(List<double> probabilities) {
  if (probabilities.isEmpty) return -1;
  double maxProb = 0.0;
  int maxIndex = 0;
  for (int i = 0; i < probabilities.length; i++) {
  if (probabilities[i] > maxProb) {
  maxProb = probabilities[i];
  maxIndex = i;
  }
  }
  return maxIndex;
  }

  Future<void> _processImageAndGetPrediction(XFile imageFile) async {
  if (_model1Interpreter == null || _model2aInterpreter == null || _model2bInterpreter == null) {
  print("Một hoặc nhiều model AI chưa được tải!");
  _showErrorSnackBar("Model AI chưa sẵn sàng. Vui lòng thử lại sau.");
  return;
  }
  if (_isProcessingImage) return;

  setState(() {
  _isProcessingImage = true;
  _predictionResultText = "Đang phân tích...";
  });

  try {
  final Uint8List imageBytes = await imageFile.readAsBytes();
  img_lib.Image? originalImage = img_lib.decodeImage(imageBytes);

  if (originalImage == null) {
  _showErrorSnackBar("Không thể xử lý ảnh này.");
  return;
  }

  ByteBuffer modelInputByteBuffer = _convertImageToByteBuffer(originalImage);

  // Chạy Model 1
  var model1OutputBuffer = List.filled(1 * 1, 0.0).reshape([1, 1]); // Output: [1,1] cho sigmoid
  _model1Interpreter!.run(modelInputByteBuffer, model1OutputBuffer);
  double model1Prob = (model1OutputBuffer[0] as List<double>)[0];

  String finalPredictionCategory;
  String finalDetailedPrediction;
  const double model1Threshold = 0.5; // Ngưỡng cho Model 1

  if (model1Prob >= model1Threshold) { // Giả sử >= threshold là 'Tái chế' (lớp 1)
  finalPredictionCategory = _model1ClassNames[1] ?? "Tái chế";
  // Chạy Model 2B (Rác Tái Chế)
  int numRecyclableClasses = _model2bClassNames.length;
  if (numRecyclableClasses == 0) {
  _showErrorSnackBar("Lỗi cấu hình lớp Model 2B"); return;
  }
  var model2bOutputBuffer = List.filled(1 * numRecyclableClasses, 0.0).reshape([1, numRecyclableClasses]);
  _model2bInterpreter!.run(modelInputByteBuffer, model2bOutputBuffer);
  List<double> model2bProbabilities = (model2bOutputBuffer[0] as List<dynamic>).cast<double>();
  int model2bPredictedIndex = _getIndexOfMax(model2bProbabilities);
  finalDetailedPrediction = _model2bClassNames[model2bPredictedIndex] ?? "Không rõ (Tái chế)";
  } else { // 'Không tái chế' (lớp 0)
  finalPredictionCategory = _model1ClassNames[0] ?? "Không tái chế";
  // Chạy Model 2A (Rác Không Tái Chế)
  int numNonRecyclableClasses = _model2aClassNames.length;
  if (numNonRecyclableClasses == 0) {
  _showErrorSnackBar("Lỗi cấu hình lớp Model 2A"); return;
  }
  var model2aOutputBuffer = List.filled(1 * numNonRecyclableClasses, 0.0).reshape([1, numNonRecyclableClasses]);
  _model2aInterpreter!.run(modelInputByteBuffer, model2aOutputBuffer);
  List<double> model2aProbabilities = (model2aOutputBuffer[0] as List<dynamic>).cast<double>();
  int model2aPredictedIndex = _getIndexOfMax(model2aProbabilities);
  finalDetailedPrediction = _model2aClassNames[model2aPredictedIndex] ?? "Không rõ (Không tái chế)";
  }

  if (mounted) {
  setState(() {
  _predictionResultText = "$finalPredictionCategory: $finalDetailedPrediction";
  });
  _showPredictionDialog(finalPredictionCategory, finalDetailedPrediction);
  }
  } catch (e) {
  print("Lỗi trong quá trình dự đoán: $e");
  _showErrorSnackBar("Lỗi dự đoán: $e");
  if (mounted) setState(() { _predictionResultText = "Lỗi dự đoán"; });
  } finally {
  if (mounted) {
  setState(() {
  _isProcessingImage = false;
  });
  }
  }
  }

  Future<void> _takePictureAndPredict() async {
  if (!_isCameraInitialized || _controller == null || !_controller!.value.isInitialized) {
  _showErrorSnackBar('Camera chưa sẵn sàng');
  return;
  }
  await _processImageAndGetPrediction(await _controller!.takePicture());
  }

  Future<void> _pickImageAndPredict() async {
  final ImagePicker picker = ImagePicker();
  final XFile? imageFile = await picker.pickImage(source: ImageSource.gallery);
  if (imageFile != null) {
  await _processImageAndGetPrediction(imageFile);
  } else {
  if (mounted) setState(() { _predictionResultText = "Chưa chọn ảnh"; });
  }
  }

  Future<void> _switchCamera() async {
  if (!_isCameraInitialized || _controller == null || cameras.isEmpty) return;

  final currentLensDirection = _controller!.description.lensDirection;
  CameraDescription newCameraDescription;

  if (currentLensDirection == CameraLensDirection.back) {
  newCameraDescription = cameras.firstWhere(
  (camera) => camera.lensDirection == CameraLensDirection.front,
  orElse: () => widget.camera // fallback to initial or current camera
  );
  } else {
  newCameraDescription = cameras.firstWhere(
  (camera) => camera.lensDirection == CameraLensDirection.back,
  orElse: () => widget.camera
  );
  }

  if (newCameraDescription.name != _controller!.description.name) {
  await _initializeCamera(newCameraDescription); // Re-initialize with new camera
  if(mounted) setState(() { _isFrontCamera = !_isFrontCamera; });
  }
  }

  void _showErrorSnackBar(String message) {
  if (mounted) {
  ScaffoldMessenger.of(context).showSnackBar(
  SnackBar(
  content: Text