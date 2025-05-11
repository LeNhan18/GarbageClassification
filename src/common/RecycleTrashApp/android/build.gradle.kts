import org.gradle.api.tasks.testing.logging.TestLogEvent
import java.nio.file.Paths
import org.gradle.api.JavaVersion
import org.gradle.jvm.toolchain.JavaLanguageVersion

buildscript {
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        classpath("com.android.tools.build:gradle:7.0.4")
    }
}

allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

// ✅ Thêm cấu hình ép Gradle dùng Java 17 (dù máy bạn cài Java 22)
java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(17))
    }
}
android {
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

}
val newBuildDir: Directory = rootProject.layout.buildDirectory.dir("../../build").get()
rootProject.layout.buildDirectory.set(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.set(newSubprojectBuildDir)
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
