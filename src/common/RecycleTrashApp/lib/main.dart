import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await _requestCameraPermission();
  final cameras = await availableCameras();
  final firstCamera = cameras.isNotEmpty ? cameras.first : null;
  runApp(MyApp(camera: firstCamera));
}

Future<void> _requestCameraPermission() async {
  try {
    PermissionStatus status = await Permission.camera.request();
    if (status.isDenied) {
      print("Camera permission denied");
      status = await Permission.camera.request();
      if (status.isDenied) {
        print("Camera permission denied again");
        return;
      }
    }
    if (status.isPermanentlyDenied) {
      print("Camera permission permanently denied");
      await openAppSettings();
    }
  } catch (e) {
    print("Error requesting camera permission: $e");
  }
}

class MyApp extends StatelessWidget {
  final CameraDescription? camera;

  const MyApp({this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Phân Loại Rác',
      theme: ThemeData(
        primarySwatch: Colors.green,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        useMaterial3: true,
      ),
      home: camera == null
          ? const NoCameraPage()
          : CameraPreviewPage(camera: camera!),
    );
  }
}

class NoCameraPage extends StatelessWidget {
  const NoCameraPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              'Không tìm thấy camera!',
              style: GoogleFonts.roboto(fontSize: 20),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () async {
                final cameras = await availableCameras();
                if (cameras.isNotEmpty) {
                  Navigator.of(context).pushReplacement(
                    MaterialPageRoute(
                      builder: (context) => CameraPreviewPage(camera: cameras.first),
                    ),
                  );
                }
              },
              child: const Text('Thử lại'),
            ),
          ],
        ),
      ),
    );
  }
}

class CameraPreviewPage extends StatefulWidget {
  final CameraDescription camera;

  const CameraPreviewPage({required this.camera});

  @override
  _CameraPreviewPageState createState() => _CameraPreviewPageState();
}

class _CameraPreviewPageState extends State<CameraPreviewPage> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  bool _isFlashOn = false;
  double _zoomLevel = 1.0;
  bool _isFrontCamera = false;
  bool _isGridVisible = false;
  List<CameraDescription> cameras = [];
  bool _isInitialized = false;
  bool _isLoading = false;
  String? _predictionResult;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    try {
      _controller = CameraController(widget.camera, ResolutionPreset.high);
      _initializeControllerFuture = _controller.initialize();
      await _initializeControllerFuture;
      if (mounted) {
        setState(() {
          _isInitialized = true;
        });
      }
      await _loadCameras();
    } catch (e) {
      print("Error initializing camera: $e");
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'Lỗi khởi tạo camera: $e',
              style: GoogleFonts.roboto(),
            ),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _loadCameras() async {
    try {
      cameras = await availableCameras();
    } catch (e) {
      print("Error loading cameras: $e");
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _sendImageForPrediction(String imagePath) async {
    setState(() {
      _isLoading = true;
      _predictionResult = null;
    });

    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('YOUR_API_ENDPOINT'), // Thay thế bằng endpoint API của bạn
      );

      request.files.add(
        await http.MultipartFile.fromPath('image', imagePath),
      );

      final response = await request.send();
      final responseData = await response.stream.bytesToString();
      final result = json.decode(responseData);

      if (mounted) {
        setState(() {
          _isLoading = false;
          _predictionResult = result['prediction'];
        });

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'Kết quả dự đoán: ${result['prediction']}',
              style: GoogleFonts.roboto(),
            ),
            backgroundColor: Colors.green,
          ),
        );
      }
    } catch (e) {
      print("Error sending image: $e");
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'Lỗi gửi ảnh: $e',
              style: GoogleFonts.roboto(),
            ),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<String> _takePicture() async {
    if (!_isInitialized || !_controller.value.isInitialized) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Camera chưa sẵn sàng'),
          backgroundColor: Colors.orange,
        ),
      );
      return '';
    }

    try {
      final XFile file = await _controller.takePicture();
      final directory = await getTemporaryDirectory();
      final String filePath = '${directory.path}/photo_${DateTime.now()}.png';
      await File(filePath).writeAsBytes(await file.readAsBytes());

      // Gửi ảnh đi dự đoán
      await _sendImageForPrediction(filePath);
      return filePath;
    } catch (e) {
      print("Error taking picture: $e");
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Lỗi chụp ảnh: $e',
            style: GoogleFonts.roboto(),
          ),
          backgroundColor: Colors.red,
        ),
      );
      return '';
    }
  }

  void _switchCamera() async {
    final lensDirection = _controller.description.lensDirection;
    CameraDescription newCamera;

    if (lensDirection == CameraLensDirection.front) {
      newCamera = cameras.firstWhere(
            (camera) => camera.lensDirection == CameraLensDirection.back,
      );
    } else {
      newCamera = cameras.firstWhere(
            (camera) => camera.lensDirection == CameraLensDirection.front,
      );
    }

    await _controller.dispose();
    _controller = CameraController(newCamera, ResolutionPreset.high);
    await _controller.initialize(); // Khởi tạo lại controller sau khi chuyển camera
    setState(() {
      _isFrontCamera = !_isFrontCamera;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [Colors.green.shade900, Colors.blue.shade900],
              ),
            ),
          ),
          FutureBuilder<void>(
            future: _initializeControllerFuture,
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.done && !snapshot.hasError) {
                return Center(
                  child: Padding(
                    padding: const EdgeInsets.all(20.0),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(30),
                      child: AspectRatio(
                        aspectRatio: _controller.value.aspectRatio,
                        child: Stack(
                          children: [
                            CameraPreview(_controller),
                            if (_isGridVisible)
                              CustomPaint(
                                size: Size.infinite,
                                painter: GridPainter(),
                              ),
                            Container(
                              decoration: BoxDecoration(
                                gradient: LinearGradient(
                                  begin: Alignment.topCenter,
                                  end: Alignment.bottomCenter,
                                  colors: [
                                    Colors.black.withOpacity(0.3),
                                    Colors.transparent,
                                    Colors.black.withOpacity(0.3),
                                  ],
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                );
              } else if (snapshot.hasError) {
                return Center(
                  child: Text(
                    'Lỗi: ${snapshot.error}',
                    style: GoogleFonts.roboto(color: Colors.white),
                  ),
                );
              }
              return const Center(
                child: CircularProgressIndicator(
                  color: Colors.white,
                  strokeWidth: 3,
                ),
              );
            },
          ),
          if (_isLoading)
            Container(
              color: Colors.black.withOpacity(0.5),
              child: const Center(
                child: CircularProgressIndicator(
                  color: Colors.white,
                ),
              ),
            ),
          if (_predictionResult != null)
            Positioned(
              top: 100,
              left: 0,
              right: 0,
              child: Container(
                margin: const EdgeInsets.symmetric(horizontal: 20),
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.7),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Text(
                  'Kết quả: $_predictionResult',
                  style: GoogleFonts.roboto(
                    color: Colors.white,
                    fontSize: 18,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
            ),
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
              child: Column(
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      IconButton(
                        icon: const Icon(Icons.arrow_back_ios, color: Colors.white),
                        onPressed: () => Navigator.pop(context),
                      ),
                      Text(
                        'Phân Loại Rác',
                        style: GoogleFonts.roboto(
                          color: Colors.white,
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      IconButton(
                        icon: Icon(
                          _isGridVisible ? Icons.grid_on : Icons.grid_off,
                          color: Colors.white,
                        ),
                        onPressed: () => setState(() => _isGridVisible = !_isGridVisible),
                      ),
                    ],
                  ),
                  Slider(
                    value: _zoomLevel,
                    min: 1.0,
                    max: 5.0,
                    activeColor: Colors.white,
                    inactiveColor: Colors.white30,
                    onChanged: _isInitialized
                        ? (value) async {
                            setState(() => _zoomLevel = value);
                            await _controller.setZoomLevel(value);
                          }
                        : null,
                  ),
                ],
              ),
            ),
          ),
          Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding: const EdgeInsets.only(bottom: 20),
              child: Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.7),
                  borderRadius: BorderRadius.circular(30),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    IconButton(
                      icon: Icon(
                        _isFlashOn ? Icons.flash_on : Icons.flash_off,
                        color: Colors.white,
                        size: 28,
                      ),
                      onPressed: _isInitialized
                          ? () async {
                              setState(() => _isFlashOn = !_isFlashOn);
                              await _controller.setFlashMode(
                                _isFlashOn ? FlashMode.torch : FlashMode.off,
                              );
                            }
                          : null,
                    ),
                    const SizedBox(width: 20),
                    Container(
                      decoration: BoxDecoration(
                        color: Colors.white,
                        shape: BoxShape.circle,
                        boxShadow: [
                          BoxShadow(
                            color: Colors.white.withOpacity(0.3),
                            spreadRadius: 2,
                            blurRadius: 5,
                          ),
                        ],
                      ),
                      child: IconButton(
                        iconSize: 35,
                        icon: const Icon(Icons.camera, color: Colors.black),
                        onPressed: _isInitialized ? _takePicture : null,
                      ),
                    ),
                    const SizedBox(width: 20),
                    IconButton(
                      icon: Icon(
                        _isFrontCamera ? Icons.camera_front : Icons.camera_rear,
                        color: Colors.white,
                        size: 28,
                      ),
                      onPressed: _isInitialized ? _switchCamera : null,
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class GridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white30
      ..strokeWidth = 1;

    for (int i = 1; i < 3; i++) {
      final x = size.width * i / 3;
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), paint);
    }

    for (int i = 1; i < 3; i++) {
      final y = size.height * i / 3;
      canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}
