# spark-workshop
cd /home/hive/spark-1.2.0.2.2.0.0-82-bin-2.6.0.2.2.0.0-2041/bin

su hive

./spark-shell --jars /home/hive/spark-streaming-twitter_2.10-1.2.0.jar,/home/hive/twitter4j-core-3.0.3.jar --num-executors 4 --executor-cores 16 --executor-memory 20g --master yarn-client



******* initial setup ******

import org.apache.spark.streaming.twitter._

import twitter4j.auth.OAuthAuthorization

import twitter4j.conf.ConfigurationBuilder

import org.apache.spark.streaming.StreamingContext

import org.apache.spark.streaming.Seconds

import org.apache.spark.mllib.linalg.Vector

import org.apache.spark.mllib.feature.HashingTF

import org.apache.spark.mllib.clustering.KMeans

val tf = new HashingTF(1000)

def featurize(s: String): Vector = {
	tf.transform(s.sliding(2).toSeq)
}

val cb = new ConfigurationBuilder()
cb.setOAuthConsumerKey("1OYIOzdy9B9pucIfz3am68jci")
cb.setOAuthConsumerSecret("6Fwm7UFzOweq8PwYHKhIKmMPzfMhYcWyVbAKoqEob7e3HXOQTm")
cb.setOAuthAccessToken("336943510-gAo2UbccZtaApq7UIlwndC0warzNjc1J3LWxgavq")
cb.setOAuthAccessTokenSecret("QoFWIJuLerryTFTO1int1wXqhlM869FqsZJpU9F68sT5e")

val auth = new OAuthAuthorization( cb.build() )


**** create orc file from the hdfs tweet data ******

val hiveCtx = new org.apache.spark.sql.hive.HiveContext(sc)

val sqlCtx = new org.apache.spark.sql.SQLContext(sc)

val tweetTableHive = hiveCtx.jsonFile("/tmp/tweets/tweets*/part-*")

tweetTableHive.printSchema()

tweetTableHive.cache()

tweetTableHive.registerTempTable("tweetTableHive")

hiveCtx.hql("select text from tweetTableHive limit 100").collect().foreach(println)

hiveCtx.hql("create table orc_tweets(user string, text string) stored as orc")

hiveCtx.hql("insert into table orc_tweets select user.name, text from tweetTableHive")

hiveCtx.hql("select user, text from orc_tweets limit 1").collect().foreach(println)




******* creating the k-means model *******

val tweetTableSql = sqlCtx.jsonFile("/tmp/tweets/tweets*/part-*").cache()

tweetTableSql.registerTempTable("tweetTableSql")

val texts = sqlCtx.sql("select text from tweetTableSql").map(_.head.toString)

val vectors = texts.map(featurize).cache() 

val model = KMeans.train(vectors, 10, 20)

val some_tweets = texts.take(100)

for (i <- 0 until 10) {
      println(s"\nCLUSTER $i:")
      some_tweets.foreach { t =>
        if (model.predict(featurize(t)) == i) {
          println(t)
        }
      }
    }


****** apply model to real time stream *****

val ssc = new StreamingContext(sc, Seconds(5))

val tweetStream = TwitterUtils.createStream(ssc, Some(auth))

val statuses = tweetStream.map(_.getText)

// select tweets matching a language cluster 
val filteredTweets = statuses.filter(t => model.predict(featurize(t)) == 3) 

