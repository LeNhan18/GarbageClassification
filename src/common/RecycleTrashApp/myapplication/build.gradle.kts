plugins {
}

    android {
    namespace = "com.example.myapplication"
    compileSdk = 35

    defaultConfig {
      applicationId = "com.example.myapplication"
    minSdk = 24
    targetSdk = 35
    versionCode = 1
    versionName = "1.0"

      testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
       release {
           isMinifyEnabled = false
           proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
       }
    }
    }

  dependencies {

  }