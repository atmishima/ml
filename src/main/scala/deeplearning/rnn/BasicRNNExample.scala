package deeplearning.rnn

import java.util.Random

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

object BasicRNNExample {

  val learnString = "Der Cottbuser Postkutscher putzt den Cottbuser Postkutschkasten.".toCharArray

  lazy val learnStringCharsList = learnString.toSet.toList

  val hiddenLayerWidth = 50
  val hiddenLayerCount = 2
  val random = new Random(7894)

  def doTrain(): Unit = {

    val builder: NeuralNetConfiguration.Builder = new NeuralNetConfiguration.Builder()
    builder.iterations(10)
    builder.learningRate(0.001)
    builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    builder.seed(123)
    builder.biasInit(0)
    builder.miniBatch(false)
    builder.updater(Updater.RMSPROP)
    builder.weightInit(WeightInit.XAVIER)

    val list: ListBuilder = builder.list()

    for (i <- 0 until hiddenLayerCount) {
      val hiddenLayerBuilder: GravesLSTM.Builder = new GravesLSTM.Builder()
      hiddenLayerBuilder.nIn(if (i == 0) learnStringCharsList.length else hiddenLayerWidth)
      hiddenLayerBuilder.nOut(hiddenLayerWidth)

      hiddenLayerBuilder.activation("tanh")
      list.layer(i, hiddenLayerBuilder.build())
    }

    val outputLayerBuilder: RnnOutputLayer.Builder = new RnnOutputLayer.Builder(LossFunction.MCXENT)
    outputLayerBuilder.activation("softmax")
    outputLayerBuilder.nIn(hiddenLayerWidth)
    outputLayerBuilder.nOut(learnStringCharsList.length)
    list.layer(hiddenLayerCount, outputLayerBuilder.build())

    // finish builder
    list.pretrain(false)
    list.backprop(true)

    // create network
    val conf: MultiLayerConfiguration = list.build()
    val net: MultiLayerNetwork = new MultiLayerNetwork(conf)
    net.init()
    net.setListeners(new ScoreIterationListener(1))

    val input: INDArray = Nd4j.zeros(1, learnStringCharsList.length, learnString.length)
    val labels: INDArray = Nd4j.zeros(1, learnStringCharsList.length, learnString.length)

    for ((c, i) <- learnString.zipWithIndex) {
      val next: Char = learnString((i+1) % learnString.length)
      input.putScalar(Array(0, learnStringCharsList.indexOf(c), i), 1)
      labels.putScalar(Array(0, learnStringCharsList.indexOf(next), i), 1)
    }

    val trainingData: DataSet = new DataSet(input, labels)

    for (epoch <- 0 until 100) {
      println(s"Epoch ${epoch}")

      net.fit(trainingData)

      net.rnnClearPreviousState()

      val testInit: INDArray = Nd4j.zeros(learnStringCharsList.length)
      testInit.putScalar(learnStringCharsList.indexOf(learnString(0)), 1)

      val output: INDArray = net.rnnTimeStep(testInit)

      testNetwork(net, output, learnStringCharsList, learnString.length)
      println()
    }

  }

  def testNetwork(net: MultiLayerNetwork, output: INDArray, learningList: List[Char], ite: Int): Any = {
    if (ite == 0) output
    else {
      val next = testNetwork(net, output, learningList)
      testNetwork(net, next, learningList, ite-1)
    }
  }

  def testNetwork(net: MultiLayerNetwork, output: INDArray, learningList: List[Char]): INDArray = {
    val outputProbDistribution: List[Double] =
      (0 until learnStringCharsList.length)
        .map(idx => output.getDouble(idx)).toList

    val sampledCharacterIdx: Int = findIndexOfHighestValue(outputProbDistribution)

    print(learnStringCharsList(sampledCharacterIdx))

    val nextInput: INDArray = Nd4j.zeros(learnStringCharsList.length)
    nextInput.putScalar(sampledCharacterIdx, 1)

    net.rnnTimeStep(nextInput)
  }

  def findIndexOfHighestValue(distribution: List[Double]): Int = {
    distribution.zipWithIndex.reduceLeft((t1, t2) => {
      if (t1._1 > t2._1) t1 else t2
    })._2
  }
}
