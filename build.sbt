// torch-rechub-scala build.sbt
// Scala 3 Recommendation System Framework using JavaCPP-PyTorch

ThisBuild / scalaVersion := "3.8.3"
ThisBuild / version := "0.1.0"
ThisBuild / organization := "torchrec"
scalacOptions += "-experimental"
fork := true
javaOptions += "--add-modules=jdk.incubator.vector"
javaOptions += "--add-exports=jdk.incubator.vector/jdk.incubator.vector=ALL-UNNAMED"

// Skip packaging for all configurations to avoid duplicate JAR entries
Compile / packageBin / skip := true
Test / packageBin / skip := true

lazy val currentPlatformClassifier = {
  val os = sys.props("os.name").toLowerCase
  val arch = sys.props("os.arch").toLowerCase
  if (os.contains("mac")) {
    if (arch.contains("aarch64") || arch.contains("arm64")) "macosx-arm64" else "macosx-x86_64"
  } else if (os.contains("linux")) {
    if (arch.contains("aarch64") || arch.contains("arm64")) "linux-arm64" else "linux-x86_64"
  } else if (os.contains("win")) {
    if (arch.contains("aarch64") || arch.contains("arm64")) "windows-arm64" else "windows-x86_64"
  } else "linux-x86_64"
}

ThisBuild / organization := "io.github.mullerhai"

// Fix java-output-version for Test configuration (sbt-typelevel defaults to 8)
Test / scalacOptions ~= (_.filterNot(s => s == "8" || s.startsWith("-java-output-version")) ++ Seq("-java-output-version:17"))
ThisBuild / organizationName := "mullerhai"
ThisBuild / startYear := Some(2025)
ThisBuild / licenses := Seq(License.Apache2)
ThisBuild / developers := List(
  Developer("mullerhai", "muller", "hai710459649@foxmail.com", url("https://github.com/mullerhe"))
)

// Sonatype / Maven Central publishing (sbt-sonatype)
//ThisBuild / tlSonatypeUseLegacyHost := false
//sonatypeCredentialHost := sonatypeCentralHost
ThisBuild / publishTo := Some("s01.oss.sonatype.org" at "https://s01.oss.sonatype.org/service/local")
publishMavenStyle := true
pomIncludeRepository := { _ => false }

//ThisBuild / javaOutputVersion := Some("21")

//val pytorchVersion = "2.12.0-1.5.14-SNAPSHOT"

resolvers += "Central Portal Snapshots" at "https://central.sonatype.com/repository/maven-snapshots/"
resolvers += "aliyun" at "https://maven.aliyun.com/repository/public"
resolvers ++= Seq(
  Resolver.sonatypeCentralSnapshots,
  "aliyun-snapshot" at "https://maven.aliyun.com/repository/public"
)
updateOptions := updateOptions.value.withLatestSnapshots(true)
resolvers += "aliyunmaven" at "https://maven.aliyun.com/repository/public"
// Source: https://mvnrepository.com/artifact/com.lihaoyi/requests
lazy val root = (project in file("."))
  .settings(
    name := "torch-rechub-scala",
    description := "Scala 3 Recommendation System Framework using JavaCPP-PyTorch",

    scalacOptions ++= Seq(
      "-deprecation",
      "-feature",
      "-unchecked",
      "-Yexplicit-nulls",
      "-language:postfixOps",
      "-language:implicitConversions"
    ),

    libraryDependencies ++= Seq(
      // JavaCPP PyTorch
//      "org.bytedeco" % "javacpp-pytorch" % pytorchVersion,
//      "org.bytedeco" % "javacpp" % "1.5.14-SNAPSHOT",
//
//      // JavaCPP CUDA support
//      "org.bytedeco" % "javacpp-cuda" % "13.1-9.19-1.5.14-SNAPSHOT",

      // Logging
      "org.slf4j" % "slf4j-api" % "2.0.16",
      "org.slf4j" % "slf4j-simple" % "2.0.16",

      // JSON parsing
//      "com.google.code.gson" % "gson" % "2.10.1",
      // Gson
      "com.google.code.gson" % "gson" % "2.14.0",

      // HTTP client
      "com.lihaoyi" %% "requests" % "0.9.3",

//      // ScalaTest
//      "org.scalatest" %% "scalatest" % "3.2.18"
      // ScalaTest for testing
      "org.scalatest" %% "scalatest" % "3.2.19" % Test
    ),

    // Disable publishing for now
    publish := {},
    publishLocal := {},

    // Some environments add duplicated resource mappings; keep jar packaging stable.
    Compile / packageBin / mappings := (Compile / packageBin / mappings).value.distinctBy(_._2),
    Test / packageBin / mappings := (Test / packageBin / mappings).value.distinctBy(_._2),

    // Additional libraryDependencies for PyTorch JavaCPP
    libraryDependencies ++= Seq(
      // PyTorch GPU
      //  "org.bytedeco" % "pytorch-platform-gpu" % "2.11.0-1.5.14",

      // JavaCPP
      "org.bytedeco" % "javacpp" % "1.5.14-SNAPSHOT",
      "org.bytedeco" % "javacpp" % "1.5.14-SNAPSHOT" classifier "linux-x86_64",

      // OpenBLAS
      "org.bytedeco" % "openblas" % "0.3.32-1.5.14-SNAPSHOT",
      "org.bytedeco" % "openblas" % "0.3.32-1.5.14-SNAPSHOT" classifier "linux-x86_64",

      // PyTorch
      "org.bytedeco" % "pytorch" % "2.12.0-1.5.14-SNAPSHOT",
      "org.bytedeco" % "pytorch" % "2.12.0-1.5.14-SNAPSHOT" classifier "linux-x86_64",

      "org.bytedeco" % "pytorch-platform-gpu" %  "2.12.0-1.5.14-SNAPSHOT" excludeAll(
        ExclusionRule("org.bytedeco", "cuda-platform"),
        ExclusionRule("org.bytedeco", "javacpp-platform"),
        ExclusionRule("org.bytedeco", "openblas-platform"),
        //      ExclusionRule("org.bytedeco", "pytorch")//,"windows-x86_64" todo do not open ，or cuda not work！！！
      ),
      // CUDA
      "org.bytedeco" % "cuda" % "13.2-9.21-1.5.14-SNAPSHOT",
      "org.bytedeco" % "cuda" % "13.2-9.21-1.5.14-SNAPSHOT" classifier "linux-x86_64",

      "org.bytedeco" % "cuda-redist" % "13.2-9.21-1.5.14-SNAPSHOT",
      "org.bytedeco" % "cuda-redist-cublas" % "13.2-9.21-1.5.14-SNAPSHOT",
      "org.bytedeco" % "cuda-redist-cudnn" % "13.2-9.21-1.5.14-SNAPSHOT",
      "org.bytedeco" % "cuda-redist-cusolver" % "13.2-9.21-1.5.14-SNAPSHOT",
      "org.bytedeco" % "cuda-redist-cusparse" % "13.2-9.21-1.5.14-SNAPSHOT",
      "org.bytedeco" % "cuda-redist-npp" % "13.2-9.21-1.5.14-SNAPSHOT",
      "org.bytedeco" % "cuda-redist-nccl" % "13.2-9.21-1.5.14-SNAPSHOT",
      "org.bytedeco" % "cuda-redist-nvcomp" % "13.2-9.21-1.5.14-SNAPSHOT",

      "org.bytedeco" % "cuda-redist" % "13.2-9.21-1.5.14-SNAPSHOT" classifier "linux-x86_64",
      "org.bytedeco" % "cuda-redist-cublas" % "13.2-9.21-1.5.14-SNAPSHOT" classifier "linux-x86_64",
      "org.bytedeco" % "cuda-redist-cudnn" % "13.2-9.21-1.5.14-SNAPSHOT" classifier "linux-x86_64",
      "org.bytedeco" % "cuda-redist-cusolver" % "13.2-9.21-1.5.14-SNAPSHOT" classifier "linux-x86_64",
      "org.bytedeco" % "cuda-redist-cusparse" % "13.2-9.21-1.5.14-SNAPSHOT" classifier "linux-x86_64",
      "org.bytedeco" % "cuda-redist-npp" % "13.2-9.21-1.5.14-SNAPSHOT" classifier "linux-x86_64",
      "org.bytedeco" % "cuda-redist-nccl" % "13.2-9.21-1.5.14-SNAPSHOT" classifier "linux-x86_64",
      "org.bytedeco" % "cuda-redist-nvcomp" % "13.2-9.21-1.5.14-SNAPSHOT" classifier "linux-x86_64",



      // OpenCV
      "org.bytedeco" % "opencv" % "4.13.0-1.5.14-SNAPSHOT",
      "org.bytedeco" % "opencv" % "4.13.0-1.5.14-SNAPSHOT" classifier "linux-x86_64",
      //  "org.bytedeco" % "opencv-platform" % "4.13.0-1.5.14-SNAPSHOT",

      // FFmpeg
      "org.bytedeco" % "ffmpeg" % "8.1-1.5.14-SNAPSHOT",
      "org.bytedeco" % "ffmpeg" % "8.1-1.5.14-SNAPSHOT" classifier "linux-x86_64",
      //  "org.bytedeco" % "ffmpeg-platform" % "8.1-1.5.14-SNAPSHOT"
      // Gson
      "com.google.code.gson" % "gson" % "2.14.0",


    )
  )

// Enable parallel execution
ThisBuild / concurrentRestrictions := Seq.empty

// Compiler settings
scalacOptions ++= Seq(
  "-WnonLocalReturn"
)

resolvers += "Central Portal Snapshots" at "https://central.sonatype.com/repository/maven-snapshots/"
resolvers += "aliyun" at "https://maven.aliyun.com/repository/public"
resolvers ++= Seq(
  Resolver.sonatypeCentralSnapshots,
  "aliyun-snapshot" at "https://maven.aliyun.com/repository/public"
)
updateOptions := updateOptions.value.withLatestSnapshots(true)
resolvers += "aliyunmaven" at "https://maven.aliyun.com/repository/public"

//libraryDependencies ++= Seq(
//  // JavaCPP
//  "org.bytedeco" % "javacpp" % "1.5.13",
//  "org.bytedeco" % "javacpp" % "1.5.13" classifier "linux-x86_64",
//
//  // OpenBLAS
//  "org.bytedeco" % "openblas" % "0.3.31-1.5.13",
//  "org.bytedeco" % "openblas" % "0.3.31-1.5.13" classifier "linux-x86_64",
//
//  // PyTorch
//  "org.bytedeco" % "pytorch" % "2.10.0-1.5.13",
//  "org.bytedeco" % "pytorch" % "2.10.0-1.5.13" classifier "linux-x86_64",
//
//  "org.bytedeco" % "pytorch-platform-gpu" % "2.10.0-1.5.13" excludeAll(
//    ExclusionRule("org.bytedeco", "cuda-platform"),
//    ExclusionRule("org.bytedeco", "javacpp-platform"),
//    ExclusionRule("org.bytedeco", "openblas-platform")
//  ),
//
//  // CUDA
//  "org.bytedeco" % "cuda" % "13.1-9.19-1.5.13",
//  "org.bytedeco" % "cuda" % "13.1-9.19-1.5.13" classifier "linux-x86_64",
//
//  "org.bytedeco" % "cuda-redist" % "13.1-9.19-1.5.13",
//  "org.bytedeco" % "cuda-redist-cublas" % "13.1-9.19-1.5.13",
//  "org.bytedeco" % "cuda-redist-cudnn" % "13.1-9.19-1.5.13",
//  "org.bytedeco" % "cuda-redist-cusolver" % "13.1-9.19-1.5.13",
//  "org.bytedeco" % "cuda-redist-cusparse" % "13.1-9.19-1.5.13",
//  "org.bytedeco" % "cuda-redist-npp" % "13.1-9.19-1.5.13",
//  "org.bytedeco" % "cuda-redist-nccl" % "13.1-9.19-1.5.13",
//  "org.bytedeco" % "cuda-redist-nvcomp" % "13.1-9.19-1.5.13",
//
//  "org.bytedeco" % "cuda-redist" % "13.1-9.19-1.5.13" classifier "linux-x86_64",
//  "org.bytedeco" % "cuda-redist-cublas" % "13.1-9.19-1.5.13" classifier "linux-x86_64",
//  "org.bytedeco" % "cuda-redist-cudnn" % "13.1-9.19-1.5.13" classifier "linux-x86_64",
//  "org.bytedeco" % "cuda-redist-cusolver" % "13.1-9.19-1.5.13" classifier "linux-x86_64",
//  "org.bytedeco" % "cuda-redist-cusparse" % "13.1-9.19-1.5.13" classifier "linux-x86_64",
//  "org.bytedeco" % "cuda-redist-npp" % "13.1-9.19-1.5.13" classifier "linux-x86_64",
//  "org.bytedeco" % "cuda-redist-nccl" % "13.1-9.19-1.5.13" classifier "linux-x86_64",
//  "org.bytedeco" % "cuda-redist-nvcomp" % "13.1-9.19-1.5.13" classifier "linux-x86_64",
//
//  // OpenCV
//  "org.bytedeco" % "opencv" % "4.13.0-1.5.13",
//  "org.bytedeco" % "opencv" % "4.13.0-1.5.13" classifier "linux-x86_64",
//
//  // FFmpeg
//  "org.bytedeco" % "ffmpeg" % "8.0.1-1.5.13",
//  "org.bytedeco" % "ffmpeg" % "8.0.1-1.5.13" classifier "linux-x86_64",
//
//
//)

// Assembly merge strategy for JavaCPP/PyTorch jars
//assemblyMergeStrategy := {
//  case _ => sbtassembly.MergeStrategy.first
//}