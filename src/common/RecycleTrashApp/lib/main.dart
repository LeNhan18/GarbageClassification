import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Yêu cầu quyền truy cập camera
  await _requestCameraPermission();

  final cameras = await availableCameras();
  final firstCamera = cameras.isNotEmpty ? cameras.first : null;

  if (firstCamera != null) {
    runApp(MyApp(camera: firstCamera));
  } else {
    print("No camera found!");
  }
}

Future<void> _requestCameraPermission() async {
  // Kiểm tra và yêu cầu quyền camera
  PermissionStatus status = await Permission.camera.request();
  if (status != PermissionStatus.granted) {
    print("Camera permission denied");
  }
}

class MyApp extends StatelessWidget {
  final CameraDescription camera;

  const MyApp({required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Camera App',
      theme: ThemeData(
        primarySwatch: Colors.deepPurple,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        useMaterial3: true,
      ),
      home: CameraPreviewPage(camera: camera),
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

  @override
  void initState() {
    super.initState();
    _controller = CameraController(widget.camera, ResolutionPreset.high);
    _initializeControllerFuture = _controller.initialize();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<String> _takePicture() async {
    try {
      await _initializeControllerFuture;
      final XFile file = await _controller.takePicture();
      final directory = await getTemporaryDirectory();
      final String filePath = '${directory.path}/photo_${DateTime.now()}.png';
      await File(filePath).writeAsBytes(await file.readAsBytes());
      return filePath;
    } catch (e) {
      print(e);
      return '';
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          Container(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [Colors.deepPurple, Colors.blueAccent],
              ),
            ),
          ),
          FutureBuilder<void>(
            future: _initializeControllerFuture,
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.done) {
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
                            Container(
                              decoration: BoxDecoration(
                                gradient: LinearGradient(
                                  begin: Alignment.topCenter,
                                  end: Alignment.bottomCenter,
                                  colors: [
                                    Colors.black.withOpacity(0.2),
                                    Colors.transparent,
                                    Colors.black.withOpacity(0.2),
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
              } else {
                return const Center(
                  child: CircularProgressIndicator(color: Colors.white),
                );
              }
            },
          ),
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  IconButton(
                    icon: const Icon(Icons.arrow_back_ios, color: Colors.white),
                    onPressed: () => Navigator.pop(context),
                  ),
                  const Text(
                    'Camera',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      fontFamily: 'Roboto',
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.settings, color: Colors.white),
                    onPressed: () {},
                  ),
                ],
              ),
            ),
          ),
          Align(
            alignment: Alignment.bottomCenter,
            child: Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.7),
                borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  IconButton(
                    icon: Icon(
                      _isFlashOn ? Icons.flash_on : Icons.flash_off,
                      color: Colors.white,
                    ),
                    onPressed: () {
                      setState(() {
                        _isFlashOn = !_isFlashOn;
                        _controller.setFlashMode(
                          _isFlashOn ? FlashMode.torch : FlashMode.off,
                        );
                      });
                    },
                  ),
                  FloatingActionButton(
                    onPressed: () async {
                      final String filePath = await _takePicture();
                      if (filePath.isNotEmpty) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            content: Text('Photo saved at: $filePath'),
                            backgroundColor: Colors.green,
                          ),
                        );
                      }
                    },
                    backgroundColor: Colors.white,
                    child: const Icon(Icons.camera_alt, color: Colors.deepPurple),
                  ),
                  IconButton(
                    icon: const Icon(Icons.zoom_in, color: Colors.white),
                    onPressed: () {
                      setState(() {
                        _zoomLevel = (_zoomLevel + 0.5).clamp(1.0, 5.0);
                        _controller.setZoomLevel(_zoomLevel);
                      });
                    },
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
