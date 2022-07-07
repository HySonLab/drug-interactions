Files and their functionalities
  + model.py: implementation of Decagon, including Bilinear/Dedicom decoder, and Decagon
  + train.py: simulate the training stage with synthesis data (randomly generated)
  + test_util.py: 
       + generate random graph for santity check
       + generate a dictionary of graph convolution with key = edge_type and value = pyg.GCNConv() 
	 (later fit in the HeteroConv of torch geometric)
  + edge_loader.py: a class named EdgeBatchIterator that generates a random edge type and a batch of edges of that edge type
		    for model to backpropagate the loss in the training stage.
	 (cloned from Decagon's github with some modifications to make it fit with the torch framework).
