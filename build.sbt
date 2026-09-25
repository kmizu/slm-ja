ThisBuild / scalaVersion := "3.3.6"
ThisBuild / organization := "io.github.kmizu"

lazy val root = (project in file("."))
  .settings(
    name := "slm-ja",
    scalacOptions ++= Seq("-deprecation", "-feature", "-Wunused:all"),
    libraryDependencies += "org.scalameta" %% "munit" % "1.1.1" % Test,
    Compile / run / fork := true,
    Compile / run / javaOptions ++= Seq("-Xmx6g", "-XX:+UseParallelGC", "--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED"),
    Test / fork := true,
    Test / javaOptions ++= Seq("--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED"),
    // 密な層の C カーネル（AVX-512 と C コンパイラがあるときだけ作る）。テストの前に作り直す
    buildNative := {
      val code = scala.sys.process.Process(Seq("scripts/build-native.sh"), baseDirectory.value).!
      if (code != 0) sys.error(s"scripts/build-native.sh が失敗した（終了コード $code）")
    },
    Test / compile := (Test / compile).dependsOn(buildNative).value
  )

lazy val buildNative = taskKey[Unit]("native/libslmkern.so を作る")
