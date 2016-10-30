package example

import java.io.PrintWriter

import org.scalatest.{FlatSpec, MustMatchers}

import scala.io.{BufferedSource, Source}

class ExampleSpec extends FlatSpec with MustMatchers{

  "for data splitting purpose" should "done" in {
    val year = 2015
    val baseDir = s"""/Users/hirakumishima/Desktop/baseball_data/${year}"""
    val source: BufferedSource = Source.fromFile(s"""${baseDir}/train.csv""")
    val targetDir = """train_repart"""

    val group: Map[String, List[(String, String, String, String)]] = source.getLines().map(line => {
      val column = line.split(",")
      val label = column(0)
      val seqKey = column(1)
      val sortKey = column(2)
      val features = column.drop(3).toList

      (seqKey, sortKey, label, features.reduce(_ + "," + _))
    }).toList.groupBy(_._1)

    group.foreach(kv => {
      println(s"write to sequence file ${kv._1}")
      val sorted = kv._2.sortBy(_._2)
      val f: List[String] = sorted.map(_._4)
      val l: List[String] = sorted.map(_._3)

      val fpw = new PrintWriter(s"${baseDir}/${targetDir}/features/part-${kv._1}")
      f.foreach(fpw.println)
      fpw.close()

      val lpw = new PrintWriter(s"${baseDir}/${targetDir}/labels/part-${kv._1}")
      l.foreach(lpw.println)
      lpw.close()
    })
  }

}
