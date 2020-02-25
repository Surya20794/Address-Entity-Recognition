# Address-Entity-Recognition

Training spaCy NER with Custom Entities

SpaCy NER already supports the entity types like- PERSONPeople, including fictional.NORPNationalities or religious or political groups.FACBuildings, airports, highways, bridges, etc.ORGCompanies, agencies, institutions, etc.GPECountries, cities, states, etc.

Our aim is to further train this model to incorporate for our own custom entities present in our dataset. To do this we have to go through the following steps-

1- Load the model, or create an empty model using spacy.blank with the ID of desired language. If a blank model is being used, we have to add the entity recognizer to the pipeline. If an existing model is being used, we have to disable all other pipeline components during training using nlp.disable_pipes. This way, only the entity recognizer gets trained.


# Setting up the pipeline and entity recognizer.
if model is not None:
    nlp = spacy.load(model)  # load existing spacy model
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.blank('en')  # create blank Language class
    print("Created blank 'en' model")
if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner)
else:
    ner = nlp.get_pipe('ner')

2-Add the new entity label to the entity recognizer using the add_label method.

 # Add new entity labels to entity recognizer
 for i in LABEL:
    ner.add_label(i)# Inititalizing optimizer
 if model is None:
    optimizer = nlp.begin_training()
 else:
    optimizer = nlp.entity.create_optimizer()



3-Loop over the examples and call nlp.update, which steps through the words of the input. At each word, it makes a prediction. It then consults the annotations, to see whether it was right. If it was wrong, it adjusts its weights so that the correct action will score higher next time.

# Get names of other pipes to disable them during training to train # only NER and update the weights

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        batches = minibatch(TRAIN_DATA,size=compounding(4., 32., 1.001))
        for batch in batches:
            texts, annotations = zip(*batch) 
            # Updating the weights
            nlp.update(texts, annotations, sgd=optimizer, 
                       drop=0.35, losses=losses)
        print('Losses', losses)            
            nlp.update(texts, annotations, sgd=optimizer, 
                       drop=0.35, losses=losses)
        print('Losses', losses)

4-Save the trained model using nlp.to_disk.

# Save model 
if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.meta['name'] = new_model_name  # rename model
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)
5-Test the model to make sure the new entity is recognized correctly.

# Test the saved model
print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)
doc2 = nlp2(test_text)
for ent in doc2.ents:
    print(ent.label_, ent.text)
