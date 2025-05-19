  import 'package:flutter/material.dart';
  import 'package:camera/camera.dart';
  import 'package:permission_handler/permission_handler.dart';
  import 'package:google_fonts/google_fonts.dart';
  import 'dart:async';

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

  class _CameraPreviewPageState extends State<CameraPreviewPage> with WidgetsBindingObserver {
    CameraController? _controller;
    Future<void>? _initializeControllerFuture;
    bool _isFlashOn = false;
    double _zoomLevel = 1.0;
    bool _isFrontCamera = false;
    bool _isGridVisible = false;
    List<CameraDescription> cameras = [];
    bool _isInitialized = false;
    bool _isCameraDisposed = false;

    @override
    void initState() {
      super.initState();
      WidgetsBinding.instance.addObserver(this);
      _initializeCamera();
    }

    @override
    void dispose() {
      WidgetsBinding.instance.removeObserver(this);
      _disposeCamera();
      super.dispose();
    }

    @override
    void didChangeAppLifecycleState(AppLifecycleState state) {
      if (!_isInitialized) return;

      // App state changed before we got official camera data
      if (_controller == null || !_controller!.value.isInitialized) return;

      if (state == AppLifecycleState.inactive) {
        _disposeCamera();
      } else if (state == AppLifecycleState.resumed) {
        _initializeCamera();
      }
    }

    Future<void> _disposeCamera() async {
      if (!_isCameraDisposed && _controller != null) {
        await _controller!.dispose();
        _isCameraDisposed = true;
        _isInitialized = false;
      }
    }

    Future<void> _initializeCamera() async {
      if (_isCameraDisposed || _controller == null) {
        _controller = CameraController(
          widget.camera,
          ResolutionPreset.high,
          enableAudio: false,
          imageFormatGroup: ImageFormatGroup.jpeg,
        );
        _isCameraDisposed = false;
      }

      try {
        _initializeControllerFuture = _controller!.initialize();
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
        if (cameras.isEmpty) {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text('Không tìm thấy camera nào'),
                backgroundColor: Colors.red,
              ),
            );
          }
        }
      } catch (e) {
        print("Error loading cameras: $e");
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Lỗi tải camera: $e'),
              backgroundColor: Colors.red,
            ),
          );
        }
      }
    }

    Future<void> _switchCamera() async {
      if (!_isInitialized || _controller == null) return;

      final lensDirection = _controller!.description.lensDirection;
      CameraDescription newCamera;

      try {
        if (lensDirection == CameraLensDirection.front) {
          newCamera = cameras.firstWhere(
                (camera) => camera.lensDirection == CameraLensDirection.back,
          );
        } else {
          newCamera = cameras.firstWhere(
                (camera) => camera.lensDirection == CameraLensDirection.front,
          );
        }

        await _disposeCamera();

        _controller = CameraController(
          newCamera,
          ResolutionPreset.high,
          enableAudio: false,
          imageFormatGroup: ImageFormatGroup.jpeg,
        );

        _initializeControllerFuture = _controller!.initialize();
        await _initializeControllerFuture;

        if (mounted) {
          setState(() {
            _isFrontCamera = !_isFrontCamera;
            _isInitialized = true;
            _isCameraDisposed = false;
          });
        }
      } catch (e) {
        print("Error switching camera: $e");
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Lỗi chuyển đổi camera: $e'),
              backgroundColor: Colors.red,
            ),
          );
        }
      }
    }

    Future<void> _takePicture() async {
      if (!_isInitialized || _controller == null) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Camera chưa sẵn sàng'),
            backgroundColor: Colors.orange,
          ),
        );
        return;
      }

      try {
        final XFile file = await _controller!.takePicture();
        if (!mounted) return;

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'Đã chụp ảnh: ${file.path}',
              style: GoogleFonts.roboto(),
            ),
            backgroundColor: Colors.green,
            duration: const Duration(seconds: 2),
          ),
        );
      } catch (e) {
        print("Error taking picture: $e");
        if (!mounted) return;

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'Lỗi chụp ảnh: $e',
              style: GoogleFonts.roboto(),
            ),
            backgroundColor: Colors.red,
          ),
        );
      }
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
            if (_initializeControllerFuture != null)
              FutureBuilder<void>(
                future: _initializeControllerFuture,
                builder: (context, snapshot) {
                  if (snapshot.connectionState == ConnectionState.done && !snapshot.hasError && _controller != null) {
                    return Center(
                      child: Padding(
                        padding: const EdgeInsets.all(20.0),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(30),
                          child: AspectRatio(
                            aspectRatio: _controller!.value.aspectRatio,
                            child: Stack(
                              fit: StackFit.expand,
                              children: [
                                CameraPreview(_controller!),
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
                    if (_controller != null)
                      Slider(
                        value: _zoomLevel,
                        min: 1.0,
                        max: 5.0,
                        activeColor: Colors.white,
                        inactiveColor: Colors.white30,
                        onChanged: _isInitialized
                            ? (value) async {
                          setState(() => _zoomLevel = value);
                          await _controller!.setZoomLevel(value);
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
                        onPressed: _isInitialized && _controller != null
                            ? () async {
                          setState(() => _isFlashOn = !_isFlashOn);
                          await _controller!.setFlashMode(
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