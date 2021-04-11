import argparse
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

def load_embeddings(args):
    print("Loading embeddings...")
    return pickle.loads(open(args["embeddings"], "rb").read())

def encode_labels(data, encoder):
    print("Encoding labels")
    return encoder.fit_transform(data["names"])

def save_model_encoder(args, rec, encoder):
    # save model
    file = open(args["recognizer"], "wb")
    file.write(pickle.dumps(rec))
    file.close()
    # save encoder
    file = open(args["le"], "wb")
    file.write(pickle.dumps(encoder))
    file.close()
    return


def main():

    # parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--embeddings", required=True,
    	help="pass the path to the facial embeddings database")
    ap.add_argument("-r", "--recognizer", required=True,
    	help="pass the output path of the trained model")
    ap.add_argument("-l", "--le", required=True,
    	help="pass the output path of the label encoder")
    args = vars(ap.parse_args())

    # load face embeddings
    data = load_embeddings(args)

    # encode names
    encoder = LabelEncoder()
    labels = encode_labels(data, encoder)

    # train the model
    kernel_type = "linear"
    print("Training SVM with " + kernel_type + " kernel...")
    rec = SVC(C=1.0, kernel=kernel_type, probability=True)
    rec.fit(data["embeddings"], labels)

    # write the model and encoder to disk
    save_model_encoder(args, rec, encoder)


main()