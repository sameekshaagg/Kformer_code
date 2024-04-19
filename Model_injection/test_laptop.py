import torch
import argparse
from fairseq import models
from fairseq.models.roberta import RobertaModel
import xml.etree.ElementTree as ET 
import string
from omegaconf import OmegaConf
from fairseq import laptop

def remove_punctuation(sentence):
    # Remove punctuation marks
    sentence = ''.join(char for char in sentence if char not in string.punctuation)
    return sentence

def sentence_to_word_list(sentence):
    # Remove punctuation marks
    sentence = remove_punctuation(sentence)
    # Split the sentence into words
    word_list = sentence.split()
    return word_list


def main():
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", default=None, type=str, required=True,
    #                     help="Path to pre-trained model")
    # parser.add_argument(
    #         "--knowledge_layer",
    #         nargs='+',
    #         default=[-1, 2],
    #         help="Layers that would add kowledge embedding",
    #     )
    # parser.add_argument(
    #         "--data_file",
    #         type=str,
    #         default='/Users/sameeksha/Desktop/dataset/Kformer/fairseq/examples/roberta/laptop/data-bin',
    #         help="The data to be evaluate",
    #     )
    # args = parser.parse_args()

    # Load XML data from file
    tree = ET.parse("/Users/sameeksha/Desktop/dataset/Sem14_data_train_filtered.xml")
    root = tree.getroot()
    nsamples =  0
    # Iterate over each <sentence> element
    for sentence_elem in root.findall('sentence'):
        # Extract text and aspect terms
        text = sentence_elem.find('text').text.strip()
        aspect_terms = []
        for aspect_term_elem in sentence_elem.find('aspectTerms').findall('aspectTerm'):
            term = aspect_term_elem.get('term')
            aspect_terms.append(term)

    roberta = RobertaModel.from_pretrained("/Users/sameeksha/Desktop/dataset/checkpoints", 'checkpoint_best.pt', '../Kformer/fairseq/examples/roberta/laptop/data_bin')
    print(roberta)
    roberta.eval()  # disable dropout
    nsamples, ncorrect = 0, 0
    with torch.no_grad():
        for sentence_elem in root.findall('sentence'):
            # Extract text and aspect terms
            text = sentence_elem.find('text').text.strip()
            aspect_terms = [aspect_term_elem.get('term') for aspect_term_elem in sentence_elem.find('aspectTerms').findall('aspectTerm')]
            polarity_label = [polarity_elem.get('polarity') for polarity_elem in sentence_elem.find('aspectTerms').findall('aspectTerm')]
            
            sentence = text
            word_list = sentence_to_word_list(sentence)

            know_bin = []
            embedding_file_path = "/Users/sameeksha/Desktop/dataset/Sem15_embeddings/hermit_ontology.embeddings.txt"  # Replace with your file path

            with open(embedding_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()  # Remove leading/trailing whitespaces
                    if line:  # Skip empty lines
                        parts = line.split(" ", 1)  # Split on the first space
                        aspect = parts[0]  # First part is the aspect
                        vector = parts[1]  # Second part is the vector
                        try:
                            vector_list = [float(num_str) for num_str in vector.split()]  # Convert string vector to list of floats
                            print("Aspect:", aspect)
                            print("Vector:", vector_list)
                        except ValueError:
                            print(f"Warning: Unable to convert vector '{vector}' to a list of floats for aspect '{aspect}'")
                            

                        for words in word_list:
                            if words == aspect:
                                tensor_vector = torch.tensor(vector_list) 
                                know_bin.append(tensor_vector)
            
            knowledge_bin = torch.stack(know_bin).unsqueeze(0)               
            for i in range(len(aspect_terms)):
                # Encode the input
                input = roberta.encode('text ' + text, 'aspect_term ' + aspect_terms[i])
                # Predict sentiment polarity
                score = roberta.predict('sentence_classification_head', input, return_logits=False)
                pred = torch.cat(score).argmax()
                print(pred)
                nsamples += 1

                if pred == polarity_label[i]:
                    ncorrect += 1

    print('Accuracy: ' + str(ncorrect / float(nsamples)))




if __name__ == "__main__":
    main()
