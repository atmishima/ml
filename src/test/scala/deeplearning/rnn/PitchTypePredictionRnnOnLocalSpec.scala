package deeplearning.rnn

import org.scalatest.{FlatSpec, MustMatchers}

class PitchTypePredictionRnnOnLocalSpec extends FlatSpec with MustMatchers {

  "train with lstm from separated sequence files" should "done" in {
    val yearTrain = 2014
    val yearTest = 2015
    val baseDir = s"""src/test/resources/mlb/"""
    val pathToTrainFeaturesFile  = s"""${baseDir}/in/${yearTrain}/features/part-%d"""
    val pathToTrainLabelsFile    = s"""${baseDir}/in/${yearTrain}/labels/part-%d"""
    val pathToTestFeaturesFile  = s"""${baseDir}/in/${yearTest}/features/part-%d"""
    val pathToTestLabelsFile    = s"""${baseDir}/in/${yearTest}/labels/part-%d"""

    PitchTypePredictionRnnOnLocal.doTrain(pathToTrainFeaturesFile, pathToTrainLabelsFile, 0, 100, pathToTestFeaturesFile, pathToTestLabelsFile, 0, 100, s"""${baseDir}/out/log.txt""", 92, 50)
  }
}
