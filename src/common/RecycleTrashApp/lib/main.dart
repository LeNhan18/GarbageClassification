import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import '_CameraPreviewPage.dart';

// Tạo một phiên bản giả lập để sử dụng trên web
bool get isUsingTFLite => !kIsWeb;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Kiểm tra nếu đang chạy trên web
  if (kIsWeb) {
    print("Đang chạy trên nền tảng web");
  } else {
    print("Đang chạy trên nền tảng native");
  }
  
  await _requestCameraPermission();
  
  try {
    final cameras = await availableCameras();
    final firstCamera = cameras.isNotEmpty ? cameras.first : null;

    if (firstCamera != null) {
      runApp(MyApp(camera: firstCamera));
    } else {
      print("Không tìm thấy camera!");
      // Hiển thị ứng dụng với thông báo lỗi
      runApp(NoCameraApp());
    }
  } catch (e) {
    print("Lỗi khi khởi tạo camera: $e");
    // Hiển thị ứng dụng với thông báo lỗi
    runApp(NoCameraApp());
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

  const MyApp({super.key, required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Phân Loại Rác AI',
      theme: ThemeData(
        primarySwatch: Colors.green,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        useMaterial3: true,
      ),
      home: CameraPreviewPage(camera: camera),
      debugShowCheckedModeBanner: false,
    );
  }
}

/// Ứng dụng hiển thị khi không tìm thấy camera
class NoCameraApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Phân loại rác',
      theme: ThemeData(
        primarySwatch: Colors.green,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: Scaffold(
        appBar: AppBar(
          title: Text('Phân loại rác'),
          backgroundColor: Colors.green,
        ),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                Icons.no_photography,
                size: 80,
                color: Colors.red,
              ),
              SizedBox(height: 20),
              Text(
                'Không thể truy cập camera',
                style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
              ),
              SizedBox(height: 10),
              Padding(
                padding: EdgeInsets.symmetric(horizontal: 40),
                child: Text(
                  'Ứng dụng cần quyền truy cập camera để phân loại rác. Vui lòng kiểm tra quyền truy cập camera hoặc thử lại trên thiết bị khác.',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 16),
                ),
              ),
              SizedBox(height: 30),
              if (kIsWeb)
                Padding(
                  padding: EdgeInsets.symmetric(horizontal: 40),
                  child: Text(
                    'Lưu ý: Tính năng phân tích rác bằng mô hình TensorFlow Lite không hoạt động trên nền tảng web. Vui lòng sử dụng ứng dụng trên thiết bị Android hoặc iOS để có trải nghiệm đầy đủ.',
                    textAlign: TextAlign.center,
                    style: TextStyle(fontSize: 14, fontStyle: FontStyle.italic, color: Colors.orange[800]),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
