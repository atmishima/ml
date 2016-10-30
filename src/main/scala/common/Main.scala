package common

import deeplearning.rnn.PitchTypePredictionRnnOnLocal

object Main {

  def main(args: Array[String]): Unit = {
    println("args :")
    args.foreach(println)
    println("********************")

    val pathToTrainFeaturesFile = args(0)
    val pathToTrainLabelsFile = args(1)
    val fromTrainIdx = args(2).toInt
    val toTrainIdx = args(3).toInt
    val pathToTestFeaturesFile = args(4)
    val pathToTestLabelsFile = args(5)
    val fromTestIdx = args(6).toInt
    val toTestIdx = args(7).toInt
    val pathToLogFile = args(8)
    val numOfFeatures = args(9).toInt
    val epochs = args(10).toInt

    PitchTypePredictionRnnOnLocal.doTrain(
      pathToTrainFeaturesFile, pathToTrainLabelsFile, fromTrainIdx, toTrainIdx,
      pathToTestFeaturesFile, pathToTestLabelsFile, fromTestIdx, toTestIdx,
      pathToLogFile, numOfFeatures, epochs)
  }
}
