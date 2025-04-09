buildscript {
    repositories {
        google()
        mavenCentral()
    }

    dependencies {
        classpath("com.android.tools.build:gradle:7.3.0")
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:1.7.10")
    }
}

allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

plugins {
    id("com.android.application") version "8.5.2" apply false
    id("org.jetbrains.kotlin.android") version "1.8.22" apply false
}

// Cấu hình build directory
val newBuildDir = layout.projectDirectory.dir("../../build")
rootProject.layout.buildDirectory.set(newBuildDir)

subprojects {
    val newSubprojectBuildDir = newBuildDir.dir(project.name)
    project.layout.buildDirectory.set(newSubprojectBuildDir)
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}

// Cấu hình cho gallery_saver
subprojects {
    if (project.name == "gallery_saver") {
        project.plugins.withType(com.android.build.gradle.BasePlugin::class) {
            project.extensions.configure<com.android.build.gradle.BaseExtension>("android") {
                namespace = "com.example.recycletrashapp"
            }
        }
    }
}

