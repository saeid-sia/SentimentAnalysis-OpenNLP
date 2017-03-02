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

  // a simple text to test the model
  val review = """i don't know what the writer's intention was with drive me crazy.
            the story is so simple my grandmother could have adapted the screenplay. """

   def main (args: Array[String]) {

     trainingModel()
     executingModel()
  }

  def trainingModel()= {
    
    //Training Dataset which is stored in a File.Each line get started with the category.In this case
    // category is positive or negative review.for example is a movie review :
    // positive peck focuses his story on familiar material that strives to give an honest portrayal of patrice lumumba...
    val trainingFile = new FileInputStream("en-movie.train")
    // The final model get saved in a File.
    val modelFile = new FileOutputStream("en-movie.model")
    // Convert each line of training Dataset to plaintext.It takes InputStreamFactory and characterSet
    val plainText = new PlainTextByLineStream( new InputStreamFactory() {
           def createInputStream: InputStream = {
         trainingFile
      }
    }, "UTF-8")
    // Parse the text and yield DocumentSample
    val docSample = new DocumentSampleStream(plainText)
    //Categorize the training dataset.First parameter is language code
    val model = DocumentCategorizerME.train("en", docSample, new TrainingParameters, new DoccatFactory)
    val modelOut = new BufferedOutputStream(modelFile)
    // Build the final model and invoke serialize method to save in a File.
    model.serialize(modelOut)
  }

  def executingModel()= {
     // read the model
    val modelIn = new FileInputStream(new File("en-movie.model"))
    //invoke the model
    val model = new DoccatModel(modelIn)
    //this class classifies the model by invoking the categorize method
    val categorizer = new DocumentCategorizerME(model)
    // pass the test doc to get categorized. it returns  an array of double which each element represents
    // the likelihood that the doc belongs to a category.
    val result  = categorizer.categorize(review)

        for( i <- 0 until categorizer.getNumberOfCategories) {
          val category = categorizer.getCategory(i)
          // prints : positive , 0.6378
          //          negative , 0.3689
          println(category + " - " + result(i))
        }
    println(categorizer.getBestCategory(result))
    println(categorizer.getAllResults(result))
      }

      // output :
      //for getBestCategory prints: negative
      //for getAllResults prints: negative[0.5871]  positive[0.4129]
}
