package deeplearning.rnn

import java.io.PrintWriter

import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.NumberedFileInputSplit
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.{Logger, LoggerFactory}

object PitchTypePredictionRnnOnLocal {

  private lazy val log: Logger = LoggerFactory.getLogger(PitchTypePredictionRnnOnLocal.getClass)

  /**
    * 隠れ層のユニット数
    */
  val hiddenLayerWidth = 64

  /**
    * SequenceRecordReaderDataSetIteratorがミニバッチとして読み込むファイル数
    */
  val miniBatchSize = 20

  /**
    * 予測対象のクラス数
    */
  val numPossibleLabels = 19

  /**
    * 回帰かどうかのフラグ
    * classificationをしたい場合にはfalseに設定する
    */
  val regression = false

  /**
    * 素性の種類
    * networkを組むときに最初のレイヤの入力ユニット数を素性と同じにしたりとなんだかんだ使うので定義しとく
    */
  var numOfFeatures = 85

  /**
    * RNNのメインエントリ
    * SequenceRecordReaderDataSetIteratorを使ってデータを読み込むのでパスの指定方法などに注意
    *  - 各ファイルには一組のsequenceのみを入れること -> データ数分だけファイルを用意する
    *  - 各ファイルには0始まりのインデックスが振られていること -> データを読むときに%dが数字に置き換えられる
    *  - 各ファイルの一行はtime stepを表す
    *
    * @param pathToTrainFeaturesFile 素性が入ったファイルのパス path/to/file/feature-%d.csv みたいにする
    * @param pathToTrainLabelsFile ラベルが入ったファイルのパス path/to/file/label-%d.csv みたいにする
    */
  def doTrain(pathToTrainFeaturesFile: String, pathToTrainLabelsFile: String, fromTrainIdx: Int, toTrainIdx: Int,
              pathToTestFeaturesFile: String, pathToTestLabelsFile: String, fromTestIdx: Int, toTestIdx: Int,
             pathToLogFile: String, numOfFeatures: Int = 85, epochs: Int = 10): Unit = {
    this.numOfFeatures = numOfFeatures

    // data set preparation
    val trainData: DataSetIterator = createDataSetIterator(pathToTrainFeaturesFile, pathToTrainLabelsFile, fromTrainIdx, toTrainIdx)
    val testData: DataSetIterator = createDataSetIterator(pathToTestFeaturesFile, pathToTestLabelsFile, fromTestIdx, toTestIdx)

    // neural network configuration
    val conf: MultiLayerConfiguration = configNetwork()

    // spark configuration
    val net: MultiLayerNetwork = new MultiLayerNetwork(conf)
    net.init()
    net.setListeners(new ScoreIterationListener(1))

    /* this listener did't work under dl4j version 0.6.0 because of 'javax.ws.rs.ProcessingException: Already connected' error */
    // net.setListeners(new HistogramIterationListener(miniBatchSize))

    val pw: PrintWriter = new PrintWriter(pathToLogFile)
    // iterate fitting and evaluation
    for (epoch <- 0 until epochs) {

      // do fitting
      net.fit(trainData); log.info(s"score at epoch ${epoch} -> score : ${net.score()}")

      val eval: Evaluation = net.evaluate(testData); println(s"evaluation at epoch ${epoch} -> accuracy : ${eval.accuracy()}   -   f1 : ${eval.f1()}")

      pw.println(s"evaluation at epoch ${epoch} -> accuracy : ${eval.accuracy()}   -   f1 : ${eval.f1()}")
      pw.println(eval.stats())
      pw.println("***********************************************************************************")
      pw.flush()

      testData.reset()
      trainData.reset()

      log.info(s"completed epoch ${epoch}")
    }

    pw.close()

    log.info("done")
  }


  /**
    * ネットワークの設定をする
    */
  def configNetwork(): MultiLayerConfiguration = {
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(1234567)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
      .weightInit(WeightInit.UNIFORM)
      .updater(Updater.NESTEROVS).momentum(0.9).learningRate(0.001).dropOut(0.2)
      .list()
      .layer(0, new GravesLSTM.Builder().name("lstm")
      .activation("relu").nIn(numOfFeatures).nOut(hiddenLayerWidth).dropOut(0.2).build())
      .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).name("output")
        .activation("softmax").nIn(hiddenLayerWidth).nOut(numPossibleLabels).build())
      .pretrain(false).backprop(true).build()

    conf
  }


  /**
    * データを読み込むための SequenceRecordReaderDataSetIterator を返す
    *
    * @param pathToFeaturesFile 素性が入ったファイルのパス path/to/file/feature-%d.csv みたいにする
    * @param pathToLabelsFile ラベルが入ったファイルのパス path/to/file/label-%d.csv みたいにする
    * @param fromIdx ファイル名につけられたインデックスの始まり
    * @param toIdx ファイル名につけられたインデックスの終わり
    *
    * @return SequenceRecordReaderDataSetIterator
    */
  def createDataSetIterator(pathToFeaturesFile: String, pathToLabelsFile: String, fromIdx: Int, toIdx: Int): DataSetIterator = {
    val featureReader = new CSVSequenceRecordReader(0, ",")
    featureReader.initialize(new NumberedFileInputSplit(pathToFeaturesFile, fromIdx, toIdx))

    val labelReader = new CSVSequenceRecordReader(0, ",")
    labelReader.initialize(new NumberedFileInputSplit(pathToLabelsFile, fromIdx, toIdx))

    val data: DataSetIterator = new SequenceRecordReaderDataSetIterator(
      featureReader,
      labelReader,
      miniBatchSize,
      numPossibleLabels,
      regression,
      SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_START)

    data
  }
}