# How does this work?

- firstly, a very primitive bigram character level model was created using counts. 
- we take the pair <c><c> of every word in the dataset (start and end is considered a special char say .).
- then we make a table of every combination possible. we fill it with the actual counts from the dataset.
- then we take a row (given a start char, what are the counts of all possible next char). we normalize it. this gives the probability.
- so to run the model, we start with . and for each iteration we pick the next letter based on a nominal distribution (meaning, the count of items follows same probability as we determined before). we stop when we reach .
- i am skipping this step in the program. its pretty clear to me.
- so i continued with the neural net version of the very same thing.