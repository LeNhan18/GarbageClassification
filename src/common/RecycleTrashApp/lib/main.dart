import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:google_fonts/google_fonts.dart';
import 'dart:async';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:image_picker/image_picker.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Yêu cầu quyền truy cập camera
  await _requestCameraPermission();

  final cameras = await availableCameras();
  final firstCamera = cameras.isNotEmpty ? cameras.first : null;

  if (firstCamera != null) {
    runApp(MyApp(cameras: cameras));
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
  final List<CameraDescription> cameras;

  const MyApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Ứng dụng Phân Loại Rác',
      theme: ThemeData(
        primarySwatch: Colors.green,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: HomePage(cameras: cameras),
    );
  }
}

class HomePage extends StatelessWidget {
  final List<CameraDescription> cameras;

  const HomePage({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Colors.green.shade900, Colors.blue.shade900],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.all(20.0),
                child: Text(
                  'Phân Loại Rác Thông Minh',
                  style: GoogleFonts.roboto(
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
              const SizedBox(height: 40),
              Expanded(
                child: GridView.count(
                  padding: const EdgeInsets.all(20),
                  crossAxisCount: 2,
                  mainAxisSpacing: 20,
                  crossAxisSpacing: 20,
                  children: [
                    _buildFeatureCard(
                      context,
                      'Chụp ảnh',
                      Icons.camera_alt,
                      Colors.blue,
                      () => Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => CameraPage(cameras: cameras),
                        ),
                      ),
                    ),
                    _buildFeatureCard(
                      context,
                      'Chọn ảnh',
                      Icons.photo_library,
                      Colors.green,
                      () => Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => ImagePickerPage(),
                        ),
                      ),
                    ),
                    _buildFeatureCard(
                      context,
                      'Hướng dẫn',
                      Icons.help_outline,
                      Colors.orange,
                      () => _showGuideDialog(context),
                    ),
                    _buildFeatureCard(
                      context,
                      'Thông tin',
                      Icons.info_outline,
                      Colors.purple,
                      () => _showInfoDialog(context),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildFeatureCard(
    BuildContext context,
    String title,
    IconData icon,
    Color color,
    VoidCallback onTap,
  ) {
    return Card(
      elevation: 8,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(20),
      ),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(20),
        child: Container(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(20),
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [color.withOpacity(0.7), color],
            ),
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                icon,
                size: 50,
                color: Colors.white,
              ),
              const SizedBox(height: 10),
              Text(
                title,
                style: GoogleFonts.roboto(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _showGuideDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(
          'Hướng dẫn sử dụng',
          style: GoogleFonts.roboto(fontWeight: FontWeight.bold),
        ),
        content: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildGuideItem('1. Chọn "Chụp ảnh" để chụp ảnh mới'),
              _buildGuideItem('2. Chọn "Chọn ảnh" để chọn ảnh từ thư viện'),
              _buildGuideItem('3. Đợi kết quả phân loại rác'),
              _buildGuideItem('4. Xem thông tin chi tiết về loại rác'),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Đóng'),
          ),
        ],
      ),
    );
  }

  Widget _buildGuideItem(String text) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.arrow_right, color: Colors.green),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              text,
              style: GoogleFonts.roboto(fontSize: 16),
            ),
          ),
        ],
      ),
    );
  }

  void _showInfoDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(
          'Thông tin ứng dụng',
          style: GoogleFonts.roboto(fontWeight: FontWeight.bold),
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Phiên bản: 1.0.0',
              style: GoogleFonts.roboto(fontSize: 16),
            ),
            const SizedBox(height: 8),
            Text(
              'Ứng dụng giúp phân loại rác thải thông minh sử dụng AI',
              style: GoogleFonts.roboto(fontSize: 16),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Đóng'),
          ),
        ],
      ),
    );
  }
}

class CameraPage extends StatefulWidget {
  final List<CameraDescription> cameras;

  const CameraPage({super.key, required this.cameras});

  @override
  _CameraPageState createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  File? _image;
  String? _prediction;
  bool _isLoading = false;

  Future<void> _takePicture() async {
    final camera = CameraController(
      widget.cameras[0],
      ResolutionPreset.medium,
    );

    await camera.initialize();

    if (!mounted) return;

    final image = await camera.takePicture();
    setState(() {
      _image = File(image.path);
    });

    await camera.dispose();
  }

  Future<void> _predictImage() async {
    if (_image == null) return;

    setState(() {
      _isLoading = true;
      _prediction = null;
    });

    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('YOUR_API_ENDPOINT_HERE'),
      );

      request.files.add(
        await http.MultipartFile.fromPath(
          'image',
          _image!.path,
        ),
      );

      var response = await request.send();
      var responseData = await response.stream.bytesToString();
      var result = json.decode(responseData);

      setState(() {
        _prediction = result['prediction'];
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _prediction = 'Lỗi: $e';
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Chụp ảnh'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (_image != null) ...[
              Image.file(
                _image!,
                height: 300,
                width: 300,
                fit: BoxFit.cover,
              ),
              const SizedBox(height: 20),
            ],
            if (_isLoading)
              const CircularProgressIndicator()
            else if (_prediction != null)
              Text(
                'Kết quả: $_prediction',
                style: const TextStyle(fontSize: 18),
              ),
            const SizedBox(height: 20),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton.icon(
                  onPressed: _takePicture,
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('Chụp ảnh'),
                ),
                const SizedBox(width: 20),
                ElevatedButton.icon(
                  onPressed: _image != null ? _predictImage : null,
                  icon: const Icon(Icons.send),
                  label: const Text('Gửi để dự đoán'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class ImagePickerPage extends StatefulWidget {
  @override
  _ImagePickerPageState createState() => _ImagePickerPageState();
}

class _ImagePickerPageState extends State<ImagePickerPage> {
  File? _image;
  String? _prediction;
  bool _isLoading = false;

  Future<void> _pickImage() async {
    final ImagePicker picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);

    if (image != null) {
      setState(() {
        _image = File(image.path);
      });
    }
  }

  Future<void> _predictImage() async {
    if (_image == null) return;

    setState(() {
      _isLoading = true;
      _prediction = null;
    });

    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('YOUR_API_ENDPOINT_HERE'),
      );

      request.files.add(
        await http.MultipartFile.fromPath(
          'image',
          _image!.path,
        ),
      );

      var response = await request.send();
      var responseData = await response.stream.bytesToString();
      var result = json.decode(responseData);

      setState(() {
        _prediction = result['prediction'];
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _prediction = 'Lỗi: $e';
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Chọn ảnh'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (_image != null) ...[
              Image.file(
                _image!,
                height: 300,
                width: 300,
                fit: BoxFit.cover,
              ),
              const SizedBox(height: 20),
            ],
            if (_isLoading)
              const CircularProgressIndicator()
            else if (_prediction != null)
              Text(
                'Kết quả: $_prediction',
                style: const TextStyle(fontSize: 18),
              ),
            const SizedBox(height: 20),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton.icon(
                  onPressed: _pickImage,
                  icon: const Icon(Icons.photo_library),
                  label: const Text('Chọn ảnh'),
                ),
                const SizedBox(width: 20),
                ElevatedButton.icon(
                  onPressed: _image != null ? _predictImage : null,
                  icon: const Icon(Icons.send),
                  label: const Text('Gửi để dự đoán'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}