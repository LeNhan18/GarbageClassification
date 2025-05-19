import com.android.build.api.variant.BuildConfigField
plugins {
    id("com.android.application")
    id("kotlin-android")
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.recycletrashapp"
    compileSdk = 35
    ndkVersion = "29.0.13113456"

    defaultConfig {
        applicationId = "com.example.recycletrashapp"
        minSdk = 21
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    buildFeatures {
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("debug")
        }
    }
}

dependencies {
    implementation("org.jetbrains.kotlin:kotlin-stdlib-jdk8:1.9.0")
    implementation("androidx.lifecycle:lifecycle-common:2.7.0")
    implementation("androidx.lifecycle:lifecycle-common-java8:2.7.0")
    implementation("androidx.lifecycle:lifecycle-process:2.7.0")
    implementation("androidx.lifecycle:lifecycle-runtime:2.7.0")
    implementation("androidx.fragment:fragment:1.7.1")
    implementation("androidx.annotation:annotation:1.8.0")
    implementation("androidx.tracing:tracing:1.2.0")
    implementation("androidx.core:core:1.13.1")
    implementation("androidx.window:window-java:1.2.0")
    implementation("com.getkeepsafe.relinker:relinker:1.4.5")
}

flutter {
    source = "../.."
}

androidComponents {
    onVariants { variant ->
        variant.buildConfigFields.put(
            "BUILD_TIME", BuildConfigField(
                "String", "\"${System.currentTimeMillis()}\"", "build timestamp"
            )
        )
    }
}