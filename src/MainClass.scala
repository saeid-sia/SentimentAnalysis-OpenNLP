/**
 * Created by Saeid Siavashi on 2/28/17.
 */

import java.io.BufferedOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStream

import opennlp.tools.doccat._
import opennlp.tools.util.InputStreamFactory
import opennlp.tools.util.PlainTextByLineStream
import opennlp.tools.util.TrainingParameters

object MainClass {

  val review = """i don't know what the writer's intention was with drive me crazy.
            the story is so simple my grandmother could have adapted the screenplay. """

   def main (args: Array[String]) {

     trainingModel()
     executingModel()
  }

  def trainingModel()= {

    val trainingFile = new FileInputStream("C:\\en-movie.train")
    val modelFile = new FileOutputStream("C:\\en-movie.model")
    val plainText = new PlainTextByLineStream( new InputStreamFactory() {
           def createInputStream: InputStream = {
         trainingFile
      }
    }, "UTF-8")
    val docSample = new DocumentSampleStream(plainText)
    val model = DocumentCategorizerME.train("en", docSample, new TrainingParameters, new DoccatFactory)
    val modelOut = new BufferedOutputStream(modelFile)
    model.serialize(modelOut)
  }

  def executingModel()= {

    val modelIn = new FileInputStream(new File("C:\\en-movie.model"))
    val model = new DoccatModel(modelIn)
    val categorizer = new DocumentCategorizerME(model)
    val result  = categorizer.categorize(review)

        for( i <- 0 until categorizer.getNumberOfCategories) {
          val category = categorizer.getCategory(i)
          println(category + " - " + result(i))
        }
    println(categorizer.getBestCategory(result))
    println(categorizer.getAllResults(result))
      }

      // output :
      // for each line in the loop prints: 1:  ... loglikelihood=-10.39720770839918	0.6
      // for getBestCategory prints: negative
      //for getAllResults prints: negative[0.5871]  positive[0.4129]
}
