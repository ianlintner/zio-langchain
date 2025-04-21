val zioVersion = "2.0.19"

ThisBuild / organization := "dev.zio"
ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "3.3.1"
ThisBuild / crossScalaVersions := Seq("3.3.1", "3.3.2", "3.4.0")
ThisBuild / scalacOptions ++= Seq(
  "-Xfatal-warnings",
  "-deprecation",
  "-unchecked",
  "-feature",
  "-language:implicitConversions",
  "-explain",
  "-source:future"
)

// Test configuration
ThisBuild / testFrameworks += new TestFramework("zio.test.sbt.ZTestFramework")

// Dependencies common to all modules
lazy val commonDependencies = Seq(
  "dev.zio" %% "zio" % zioVersion,
  "dev.zio" %% "zio-streams" % zioVersion,
  "dev.zio" %% "zio-json" % "0.6.2",
  "dev.zio" %% "zio-config" % "4.0.0-RC16",
  "dev.zio" %% "zio-config-typesafe" % "4.0.0-RC16",
  "dev.zio" %% "zio-http" % "3.0.0-RC2",
  "dev.zio" %% "zio-cache" % "0.2.3",
  "dev.zio" %% "zio-logging" % "2.1.16",
  "dev.zio" %% "zio-nio" % "2.0.2",
  "dev.zio" %% "zio-test" % zioVersion % Test,
  "dev.zio" %% "zio-test-sbt" % zioVersion % Test,
  "dev.zio" %% "zio-test-magnolia" % zioVersion % Test
)

// Common settings for all projects
lazy val commonSettings = Seq(
  scalaVersion := "3.3.1",
  scalacOptions ++= Seq(
    "-Xfatal-warnings",
    "-deprecation",
    "-unchecked",
    "-feature",
    "-language:implicitConversions"
  ),
  Compile / doc / scalacOptions ++= Seq(
    "-groups",
    "-sourcepath", (LocalRootProject / baseDirectory).value.getAbsolutePath,
    "-doc-root-content", (Compile / resourceDirectory).value + "/rootdoc.txt"
  )
)

lazy val root = project
  .in(file("."))
  .settings(
    name := "zio-langchain",
    publish / skip := true
  )
  .aggregate(
    core,
    models,
    embeddings,
    memory,
    documentLoaders,
    documentParsers,
    documentProcessors,
    parsers,
    retrievers,
    chains,
    agents,
    tools,
    rag,
    evaluation,
    integrationOpenAI,
    integrationAnthropic,
    integrationPinecone,
    examples
  )

lazy val core = project
  .in(file("modules/core"))
  .settings(
    name := "zio-langchain-core",
    libraryDependencies ++= commonDependencies
  )

lazy val models = project
  .in(file("modules/models"))
  .dependsOn(core)
  .settings(
    name := "zio-langchain-models",
    libraryDependencies ++= commonDependencies
  )

lazy val embeddings = project
  .in(file("modules/embeddings"))
  .dependsOn(core)
  .settings(
    name := "zio-langchain-embeddings",
    libraryDependencies ++= commonDependencies
  )

lazy val memory = project
  .in(file("modules/memory"))
  .dependsOn(core)
  .settings(
    name := "zio-langchain-memory",
    libraryDependencies ++= commonDependencies ++ Seq(
      // Using core langchain4j - memory implementation is custom
      "dev.zio" %% "zio-cache" % "0.2.3"
    )
  )

lazy val documentLoaders = project
  .in(file("modules/document-loaders"))
  .dependsOn(core)
  .settings(
    name := "zio-langchain-document-loaders",
    libraryDependencies ++= commonDependencies
  )

lazy val documentParsers = project
  .in(file("modules/document-parsers"))
  .dependsOn(core)
  .settings(
    name := "zio-langchain-document-parsers",
    libraryDependencies ++= commonDependencies
  )

lazy val documentProcessors = project
  .in(file("modules/document-processors"))
  .dependsOn(core)
  .settings(
    name := "zio-langchain-document-processors",
    libraryDependencies ++= commonDependencies
  )

lazy val retrievers = project
  .in(file("modules/retrievers"))
  .dependsOn(core, embeddings)
  .settings(
    name := "zio-langchain-retrievers",
    libraryDependencies ++= commonDependencies
  )

lazy val rag = project
  .in(file("modules/rag"))
  .dependsOn(core, models, retrievers)
  .settings(
    name := "zio-langchain-rag",
    libraryDependencies ++= commonDependencies
  )

lazy val chains = project
  .in(file("modules/chains"))
  .dependsOn(core, models, retrievers)
  .settings(
    name := "zio-langchain-chains",
    libraryDependencies ++= commonDependencies
  )

lazy val agents = project
  .in(file("modules/agents"))
  .dependsOn(core, models, tools, chains)
  .settings(
    name := "zio-langchain-agents",
    libraryDependencies ++= commonDependencies
  )

lazy val tools = project
  .in(file("modules/tools"))
  .dependsOn(core)
  .settings(
    name := "zio-langchain-tools",
    libraryDependencies ++= commonDependencies
  )

lazy val parsers = project
  .in(file("modules/parsers"))
  .dependsOn(core)
  .settings(
    name := "zio-langchain-parsers",
    libraryDependencies ++= commonDependencies
  )

lazy val integrationOpenAI = project
  .in(file("modules/integrations/openai"))
  .dependsOn(core, models, embeddings)
  .settings(
    name := "zio-langchain-openai",
    libraryDependencies ++= commonDependencies
  )

lazy val integrationAnthropic = project
  .in(file("modules/integrations/anthropic"))
  .dependsOn(core, models)
  .settings(
    name := "zio-langchain-anthropic",
    libraryDependencies ++= commonDependencies
  )

lazy val integrationPinecone = project
  .in(file("modules/integrations/pinecone"))
  .dependsOn(core, models, embeddings, retrievers)
  .settings(
    name := "zio-langchain-pinecone",
    libraryDependencies ++= commonDependencies
  )

lazy val evaluation = project
  .in(file("modules/evaluation"))
  .dependsOn(core, models, retrievers, chains)
  .settings(
    name := "zio-langchain-evaluation",
    libraryDependencies ++= commonDependencies
  )

// Commented out as langchain4j-huggingface dependency may not be compatible with chosen version
// lazy val integrationHuggingFace = project
//   .in(file("modules/integrations/huggingface"))
//   .dependsOn(core, models, embeddings)
//   .settings(
//     name := "zio-langchain-huggingface",
//     libraryDependencies ++= commonDependencies
//   )

lazy val examples = project
  .in(file("modules/examples"))
  .dependsOn(
    core,
    models,
    embeddings,
    chains,
    agents,
    integrationOpenAI,
    integrationAnthropic,
    integrationPinecone,
    documentLoaders,
    documentParsers,
    documentProcessors,
    parsers,
    retrievers,
    rag,
    memory,
    tools,
    evaluation
  )
  .settings(
    name := "zio-langchain-examples",
    libraryDependencies ++= commonDependencies,
    publish / skip := true
  )

// Settings for publishing
ThisBuild / publishMavenStyle := true
ThisBuild / publishTo := {
  val nexus = "https://s01.oss.sonatype.org/"
  if (isSnapshot.value)
    Some("snapshots" at nexus + "content/repositories/snapshots")
  else
    Some("releases" at nexus + "service/local/staging/deploy/maven2")
}
ThisBuild / pomExtra :=
  <url>https://github.com/zio/zio-langchain</url>
  <licenses>
    <license>
      <name>Apache License, Version 2.0</name>
      <url>https://www.apache.org/licenses/LICENSE-2.0</url>
    </license>
  </licenses>
  <scm>
    <url>git@github.com:zio/zio-langchain.git</url>
    <connection>scm:git:git@github.com:zio/zio-langchain.git</connection>
  </scm>
  <developers>
    <developer>
      <id>zio</id>
      <name>ZIO Contributors</name>
      <url>https://github.com/zio/zio-langchain/graphs/contributors</url>
    </developer>
  </developers>