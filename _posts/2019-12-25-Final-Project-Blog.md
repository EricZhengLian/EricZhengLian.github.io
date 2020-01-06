---layout: post
Title: Final Project Daily Log---

Below is a log of my daily progress, including:
1.) What I worked on today
2.) What I learned
3.) Any road blocks or questions that I need to get answered
4.) What my goals are for tomorrow
January 7th
1.) I worked on researching relevant information about music generation through deep learning, in pariticular, the application of LSTM-RNN (Long-short-term memory recurrent-net-work) in this field. I also began to learn the basics of deep learning by watching all videos in Andrew Ng's Sequential Model section of his deep learning coursera course.
2.) I learned the mechanisms of RNN: inputting the information yielded from previous timesteps through various different "states" (i.e. cell states) and hyperparameters to the next timestep so that temporal characteristics and relationship are preserved during the training/learning process. In the fundamental example of text generation (text-level or character-level), unique elements in the sequence form the "vocabulary" (it sounds more like a glossary to me) of the inputs, from which one can one-hot encode the sequences to proccessible matrices to feed into the model. There is also a loss function, which is a measure of error between the predicted and target outputs. Usually, the target outputs are the same as inputs except they are one step ahead. For instance, when one attempts to train the RNN to learn the word "hello" with a example sequence of length 4, he would input the one-hot encoded version of 'h e l l' and expect the model to successfully predict 'e l l o' at corresponding timesteps ('successful' means the probability of the target output is the highest within the probability distribution that the model actually spits out). The loss function is usually minimized by stochastic gradient descent or batch gradient descent, which is in between the regular gradient descent and stochastic gradient descent. In the calculation of gradient at each timestep, the expansion of chain rule will involve all the previous parameters, at a high level, propagating information back all the way to the very beginning. This is called "backpropagation". However, backpropagation can generate issues such as the vanishing gradient since a long product derivatives whose absolute value is less than one will converge to 0. LSTM comes in handy here to mitigate the risk of vanishing gradient by its mechanisms of three different gates, which altogether operate to selectively forget and remember information from the past for a relatively long period of time. 
3.) Terminologies such as "epoch", "batch-size", "activation", "cross entropy" and "soft max".
4,) Goals for tomorrow: learn more about LSTM and specific python libraries which can enforce it. Start to plan on the blue prints of the project (like a method section).
January 9th
January 10th
January 13th
January 15th
January 16th
January 17th
