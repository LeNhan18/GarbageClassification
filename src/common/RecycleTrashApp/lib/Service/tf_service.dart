import 'dart:io';
import 'dart:async';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'tf_service_interface.dart';
import 'tf_service_native.dart';
import 'tf_service_web.dart';

// Factory class that returns the appropriate implementation based on platform
class TFLiteService implements TFLiteServiceInterface {
  late TFLiteServiceInterface _implementation;
  
  // Singleton pattern
  static final TFLiteService _instance = TFLiteService._internal();
  factory TFLiteService() => _instance;
  
  TFLiteService._internal() {
    // Select the appropriate implementation based on platform
    if (kIsWeb) {
      _implementation = TFLiteServiceWeb();
    } else {
      _implementation = TFLiteServiceNative();
    }
  }
  
  @override
  Future<void> loadModels() async {
    return _implementation.loadModels();
  }
  
  @override
  Future<List<double>> classifyImage(File imageFile, int modelNumber) async {
    return _implementation.classifyImage(imageFile, modelNumber);
  }
  
  @override
  void dispose() {
    _implementation.dispose();
  }
}