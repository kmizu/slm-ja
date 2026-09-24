ThisBuild / scalaVersion := "3.3.6"
ThisBuild / organization := "io.github.kmizu"

lazy val root = (project in file("."))
  .settings(
    name := "slm-ja-1m",
    scalacOptions ++= Seq("-deprecation", "-feature", "-Wunused:all"),
    libraryDependencies += "org.scalameta" %% "munit" % "1.1.1" % Test,
    Compile / run / fork := true,
    Compile / run / javaOptions ++= Seq("-Xmx6g", "-XX:+UseParallelGC", "--add-modules=jdk.incubator.vector"),
    Test / fork := true,
    Test / javaOptions += "--add-modules=jdk.incubator.vector"
  )
