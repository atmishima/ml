import sbt.{ExclusionRule, _}

object Version {
  val scala          = "2.10.6"
  val scalaTest      = "3.0.0"
  val scalaMock      = "3.2.2"
  val dl4j           = "0.6.0"
  val nd4j           = "0.6.0"
}

object Library {
  val scalaTest      = "org.scalatest"     %% "scalatest"        % Version.scalaTest % "test"
  val scalaMock      = "org.scalamock"     %% "scalamock-scalatest-support"   % Version.scalaMock % "test"

  val dl4j           = "org.deeplearning4j" % "deeplearning4j-core" % Version.dl4j
  val dl4jUi         = "org.deeplearning4j" % "deeplearning4j-ui" % Version.dl4j
  val dl4jNlp        = "org.deeplearning4j" % "deeplearning4j-nlp" % Version.dl4j
  val nd4jNativePlatform = "org.nd4j" % "nd4j-native-platform" % Version.nd4j
  val nd4j           =  "org.nd4j" % "nd4j-native" % Version.nd4j classifier "" classifier "macosx-x86_64"
  //  val nd4j           =  "org.nd4j" % "nd4j-native" % Version.nd4j classifier "" classifier "linux-x86_64"
}

object Dependencies {

  import Library._

  val baseballDl4j = List (
    scalaTest,
    dl4j,
    dl4jUi,
    dl4jNlp,
    nd4jNativePlatform,
    nd4j
  )
}
