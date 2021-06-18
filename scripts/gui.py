try:
    import tkinter as tk                # python 3
    from tkinter import font as tkfont  # python 3
except ImportError:
    import Tkinter as tk     # python 2
    import tkFont as tkfont  # python 2

from tkinter import *
from tkmacosx import Button
from tkinter import filedialog, Text, font
import os
from preprocessing_tweets import preprocess_data, lemmatizer
from gensim.corpora import Dictionary
import pickle
import gensim
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from preprocessing_tweets import ngrams, make_bow_corpus
from pprint import pprint
import pyLDAvis.gensim
import pyLDAvis
from gensim.models import LdaModel

HEIGHT = 500
WIDTH = 500
files = []

"""
lets you choose a csv file and adds it to save.txt
"""
def addFile(frame):

    for widget in frame.winfo_children():
        widget.destroy()

    filename = filedialog.askopenfilename(initialdir="/", title="Select File", filetypes=(("CSV-Files", "*.csv"), ("all files", "*.*")))
    files.append(filename)
    print(filename)
    with open('save.txt', 'w') as f:
        for file in files:
            f.write(file + '\n')
    for file in files:
        label = Label(frame, text="You selected:\n"+file, bg="white")
        label.place(relwidth=1, relheight=1)

"""
checks for valid user entries for the parameters
:return: message, validity
"""
def arevalid(topics, below, above, chunksize, passes, iterations):
    m = ""
    valid = False

    try:
        int(iterations)
        if int(iterations) <= 0:
            m = "Iterations has to be a positive integer"
            valid6 = False
        else:
            valid6 = True
    except:

        m = "Iterations has to be an integer"
        valid6 = False

    try:
        int(passes)
        if int(passes) <= 0:
            m = "Passes has to be a positive integer"
            valid5 = False
        else:
            valid5 = True
    except:

        m = "Passes has to be an integer"
        valid5 = False

    try:
        int(chunksize)
        if int(chunksize) <= 0:
            m = "Chunk Size has to be a positive integer"
            valid4 = False
        else:
            valid4 = True
    except:

        m = "Chunk Size has to be an integer"
        valid4 = False

    try:
        float(above)
        if float(above) < 0.0 or float(above) > 1.0:
            m = "No_above has to be a number \nbetween 0 and 1"
            valid1 = False
        else:
            valid1 = True
    except:

        m = "No_above has to be a number between 0 and 1"
        valid1 = False

    try:
        int(below)
        if int(below) < 0:
            m = "No_below has to be a positive integer"
            valid2 = False
        else:
            valid2 = True
    except:

        m = "No_below has to be an integer"
        valid2 = False

    try:
        int(topics)
        if int(topics)<=0:
            m = "Nr. of Topics has to be a positive integer"
            valid3 = False
        else:
            valid3 = True
    except:

        m = "Nr. of Topics has to be an integer"
        valid3 = False


    return m, valid1&valid2&valid3&valid4&valid5&valid6

"""
deletes saved data so you can use a new corpus
(currently commented)
"""
def delete():
    pass
    """
    os.remove(("../data/docs")
    """

"""
Mostly equivalent to run_preprocessing.py (Check the paths)
If preprocessing was already done (meaning 'docs' file exists) nothing happens
Otherwise the preprocessing will be done on the selected files (listed in 'save.txt')
"""
def preprocess(file):
    if not os.path.exists("../data/docs"):
        file_list = []
        with open(file, "r") as f:
            for x in f.readlines():
                x = x.replace("\n", "")
                file_list.append(x)

        docs = preprocess_data(input_file_list=file_list)
        docs = lemmatizer(input_docs=docs)
        """ Save docs list """
        with open('../data/docs', 'wb') as f:
            pickle.dump(docs, f)

"""
Mostly equivalent to run_tm.py (Check the paths)
Gets parameters from the user
If the parameters aren't valid an error message pops up
"""
def run_tm(topics, below, above, chunksize, passes, iterations):

    m, valid = arevalid(topics, below, above, chunksize, passes, iterations)
    if not valid:

        fehlerfenster = Toplevel()
        fehlerfenster.title('Fehler')
        fehlerfenster.geometry('300x300')
        # Label mit der Fehlermeldung
        labelfehler = Label(master=fehlerfenster, text=m)
        labelfehler.place(x=10, y=10, width=300, height=300)

    else:

        with open('../data/docs', 'rb') as f:
            docs = pickle.load(f)

        tweet_dictionary = Dictionary(docs)
        tweet_dictionary.filter_extremes(no_below=int(below), no_above=float(above))
        tweet_dictionary.save('../data/tweet_dictionary')

        ngram_docs = ngrams(input_docs=docs)
        corpus = make_bow_corpus(tweet_dictionary, ngram_docs)
        with open('../data/bow_corpus', 'wb') as f:
            pickle.dump(corpus, f)
        print('Number of unique tokens: %d' % len(tweet_dictionary))
        print('Number of documents: %d' % len(corpus))
        """Training parameters."""
        num_topics = int(topics)  # Number of topics, here relatively low so we can interpret them more easily -> can be set higher
        chunk_size = int(chunksize)  # Numbers of documents fed into the training algorithm (we have 7)
        passes = int(passes)  # Number of times trained on the entire corpus
        iterations = int(iterations)  # Number of loops over each document
        eval_every = None  # Don't evaluate model perplexity, takes too much time.

        """ Make a index to word dictionary."""
        temp = tweet_dictionary[0]  # This is only to "load" the dictionary.
        id2word = tweet_dictionary.id2token

        """Create model
        We set alpha = 'auto' and eta = 'auto'. Again this is somewhat technical, but essentially we are automatically learning
        two parameters in the model that we usually would have to specify explicitly."""
        model = LdaModel(
            corpus=corpus,
            id2word=id2word,
            chunksize=chunk_size,
            alpha='auto',
            eta='auto',
            iterations=iterations,
            num_topics=num_topics,
            passes=passes,
            eval_every=eval_every
        )
        model_file = '../data/model/LDA_model_v1'
        model.save(model_file)
        """ Tests """
        # Top topics
        top_topics = model.top_topics(corpus)  # , num_words=20) Default value = 20, input is our corpus in BOW format

        # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
        """Topic Coherence measures score a single topic by measuring the degree of semantic similarity between high scoring 
        words in the topic. These measurements help distinguish between topics that are semantically interpretable topics and 
        topics that are artifacts of statistical inference """
        avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
        print('Average topic coherence: %.4f.' % avg_topic_coherence)

        pprint(top_topics)

"""
Equivalent to visualization.py (Check the paths)
"""
def visualize():
    with open('../data/bow_corpus', 'rb') as input_file:
        corpus = pickle.load(input_file)

    tweet_dictionary = gensim.corpora.Dictionary.load('../data/tweet_dictionary')

    """ Load model """
    model = LdaModel.load('../data/model/LDA_model_v1')

    """ Visualization """

    lda_visualization = pyLDAvis.gensim.prepare(model, corpus, tweet_dictionary)
    pyLDAvis.show(lda_visualization)

"""
For organising the frames, enables switching between them 
"""
class OrganiseFrames(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("LDA")
        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")

        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, ChooseCorpus, Preprocess, SelectParams, ShowTopics, End):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()

"""
This is the StartPage with the ReadMe
"""
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        canvas = Canvas(self, height=HEIGHT, width=WIDTH)
        canvas.pack()

        background_frame = Frame(self, bg="#7ECFEC")
        background_frame.place(relwidth=1, relheight=1)

        upper_frame = Frame(self, bg="Black")
        upper_frame.place(relx=0.5, rely=0.02, relwidth=0.9, relheight=0.07, anchor='n')
        welcomelabel = Label(upper_frame, text="Welcome!", fg="Black", font=("Modern", 15, "bold"))
        welcomelabel.place(relwidth=1, relheight=1)

        middle_frame = Frame(self, bg='Black')
        middle_frame.place(relx=0.5, rely=0.1, relwidth=0.9, relheight=0.8, anchor='n')
        textlabel = Label(middle_frame, text="This GUI will help you to perform LDA Topic Modeling. \n \nFirst you will be asked "
                                             "to select your (german) corpus.\n \n  After that you can specify the parameters, e.g. the "
                                             "number of topics. \n \n \n Click  'Got it'  to continue. \n \n Good luck!",
                          bg='white', font=('Modern', 12))
        textlabel.place(relwidth=1, relheight=1)

        lower_frame = Frame(self, bg='Black', bd=5)
        lower_frame.place(relx=0.5, rely=0.92, relwidth=0.2, relheight=0.07, anchor='n')
        button = Button(lower_frame, text='Got it!', bg='white', fg='black', borderless=0, font=40,
                        command=lambda: controller.show_frame("ChooseCorpus"))
        button.place(relx=0, relwidth=1, relheight=1)

"""
Here you can choose the corpus if you haven't already done so
"""
class ChooseCorpus(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        canvas = Canvas(self, height=HEIGHT, width=WIDTH)
        canvas.pack()

        background_frame = Frame(self, bg="#7ECFEC")
        background_frame.place(relwidth=1, relheight=1)

        upper_frame = Frame(self, bg="Black")
        upper_frame.place(relx=0.5, rely=0.02, relwidth=0.9, relheight=0.07, anchor='n')
        welcomelabel = Label(upper_frame, text="Corpus", fg="Black", font=("Modern", 15, "bold"))
        welcomelabel.place(relwidth=1, relheight=1)

        middle_frame = Frame(self, bg='Black')
        middle_frame.place(relx=0.5, rely=0.1, relwidth=0.9, relheight=0.8, anchor='n')
        textlabel = Label(middle_frame, text="If this is your first time performing LDA, \n please select some csv files"
                                             " to build your corpus.\n Click 'Select File' to do so. \n \n"
                                             "Otherwise your saved data will be used, \n so you can skip this step and the preprocessing.", bg='white', font=('Modern', 12))
        textlabel.place(relwidth=1, relheight=1)

        lower_frame = Frame(self, bg='Black', bd=5)
        lower_frame.place(relx=0.5, rely=0.92, relwidth=0.5, relheight=0.07, anchor='n')
        button1 = Button(lower_frame, text='Select File', bg='white', fg='black', borderless=0, font=40,
                        command=lambda: addFile(middle_frame))
        button1.place(relx=0, relwidth=0.5, relheight=1)
        button2 = Button(lower_frame, text='Done', bg='white', fg='black', borderless=0, font=40,
                        command=lambda: controller.show_frame("Preprocess"))
        button2.place(relx=0.5, relwidth=0.5, relheight=1)

"""
For preprocessing the data
"""
class Preprocess(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        canvas = Canvas(self, height=HEIGHT, width=WIDTH)
        canvas.pack()

        background_frame = Frame(self, bg="#7ECFEC")
        background_frame.place(relwidth=1, relheight=1)

        upper_frame = Frame(self, bg="Black")
        upper_frame.place(relx=0.5, rely=0.02, relwidth=0.9, relheight=0.07, anchor='n')
        welcomelabel = Label(upper_frame, text="Preprocessing", fg="Black", font=("Modern", 15, "bold"))
        welcomelabel.place(relwidth=1, relheight=1)

        middle_frame = Frame(self, bg='White')
        middle_frame.place(relx=0.5, rely=0.1, relwidth=0.9, relheight=0.8, anchor='n')
        label1 = Label(middle_frame, text = "You can now preprocess your selected data.\n \n If this is your first time "
                                            "preprocessing,\n this might take some time.\n \n Otherwise your saved data "
                                            "will be loaded. \n If this is the case, you can skip this step.")
        label1.place(relx=0., rely=0.1, relwidth=1, relheight=0.5)
        button = Button(middle_frame, text="Click here to preprocess", bg='#7ECFEC', fg='black', borderless=0, font=40,
                        command=lambda: preprocess("./save.txt"))
        button.place(relx=0.1, rely=0.7, relwidth=0.8, relheight=0.2)

        lower_frame = Frame(self, bg='Black', bd=5)
        lower_frame.place(relx=0.5, rely=0.92, relwidth=0.5, relheight=0.07, anchor='n')
        button1 = Button(lower_frame, text='Go Back', bg='white', fg='black', borderless=0, font=40,
                         command=lambda: controller.show_frame("ChooseCorpus"))
        button1.place(relx=0, relwidth=0.5, relheight=1)
        button2 = Button(lower_frame, text='Done', bg='white', fg='black', borderless=0, font=40,
                         command=lambda: controller.show_frame("SelectParams"))
        button2.place(relx=0.5, relwidth=0.5, relheight=1)

"""
For selecting the parameters of the model
"""
class SelectParams(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        canvas = Canvas(self, height=HEIGHT, width=WIDTH)
        canvas.pack()

        background_frame = Frame(self, bg="#7ECFEC")
        background_frame.place(relwidth=1, relheight=1)

        upper_frame = Frame(self, bg="Black")
        upper_frame.place(relx=0.5, rely=0.02, relwidth=0.9, relheight=0.07, anchor='n')
        welcomelabel = Label(upper_frame, text="Parameters", fg="Black", font=("Modern", 15, "bold"))
        welcomelabel.place(relwidth=1, relheight=1)

        middle_frame = Frame(self, bg='White')
        middle_frame.place(relx=0.5, rely=0.1, relwidth=0.9, relheight=0.8, anchor='n')
        label = Label(middle_frame, text="Select your parameters and click 'Run the model'. \n No_above needs to be a float"
                                         " between 0 and 1. \n The other parameters should be integers. ")
        label.place(rely=0.05, relwidth=1, relheight=0.2)

        label1 = Label(middle_frame, text = "Number of Topics:")
        label1.place(rely=0.25, relwidth=0.5, relheight=0.1)
        entry1 = Entry(middle_frame)
        entry1.place(relx=0.5,rely=0.25, relwidth=0.3, relheight=0.1)
        entry1.insert(END, '10')
        entry1.place(relx=0.5,rely=0.25, relwidth=0.3, relheight=0.1)

        label2 = Label(middle_frame, text="No Below:")
        label2.place(rely=0.35, relwidth=0.5, relheight=0.1)
        entry2 = Entry(middle_frame)
        entry2.place(relx=0.5, rely=0.35, relwidth=0.3, relheight=0.1)
        entry2.insert(END, '1')
        entry2.place(relx=0.5, rely=0.35, relwidth=0.3, relheight=0.1)

        label3 = Label(middle_frame, text="No Above:")
        label3.place(rely=0.45, relwidth=0.5, relheight=0.1)
        entry3 = Entry(middle_frame)
        entry3.place(relx=0.5, rely=0.45, relwidth=0.3, relheight=0.1)
        entry3.insert(END, '0.5')
        entry3.place(relx=0.5, rely=0.45, relwidth=0.3, relheight=0.1)

        label4 = Label(middle_frame, text="Chunk Size:")
        label4.place(rely=0.55, relwidth=0.5, relheight=0.1)
        entry4 = Entry(middle_frame)
        entry4.place(relx=0.5, rely=0.55, relwidth=0.3, relheight=0.1)
        entry4.insert(END, '2000')
        entry4.place(relx=0.5, rely=0.55, relwidth=0.3, relheight=0.1)

        label5 = Label(middle_frame, text="Passes:")
        label5.place(rely=0.65, relwidth=0.5, relheight=0.1)
        entry5 = Entry(middle_frame)
        entry5.place(relx=0.5, rely=0.65, relwidth=0.3, relheight=0.1)
        entry5.insert(END, '20')
        entry5.place(relx=0.5, rely=0.65, relwidth=0.3, relheight=0.1)

        label6 = Label(middle_frame, text="Iterations:")
        label6.place(rely=0.75, relwidth=0.5, relheight=0.1)
        entry6 = Entry(middle_frame)
        entry6.place(relx=0.5, rely=0.75, relwidth=0.3, relheight=0.1)
        entry6.insert(END, '400')
        entry6.place(relx=0.5, rely=0.75, relwidth=0.3, relheight=0.1)

        low_frame = Frame(self, bg='White', bd=5)
        low_frame.place(relx=0.5, rely=0.8, relwidth=0.3, relheight=0.07, anchor='n')
        button = Button(low_frame, text='Run the model', bg='#7ECFEC', fg='black', borderless=0, font=40,
                         command=lambda: run_tm(entry1.get(), entry2.get(), entry3.get(), entry4.get(), entry5.get(), entry6.get()))
        button.place(relwidth=1, relheight=1)
        lower_frame = Frame(self, bg='Black', bd=5)
        lower_frame.place(relx=0.5, rely=0.92, relwidth=0.5, relheight=0.07, anchor='n')
        button1 = Button(lower_frame, text='Go Back', bg='white', fg='black', borderless=0, font=40,
                         command=lambda: controller.show_frame("Preprocess"))
        button1.place(relx=0, relwidth=0.5, relheight=1)
        button2 = Button(lower_frame, text='Done', bg='white', fg='black', borderless=0, font=40,
                         command=lambda: controller.show_frame("ShowTopics"))
        button2.place(relx=0.5, relwidth=0.5, relheight=1)

"""
Clicking the button in ShowTopics will open the pyLDAvis visulization
"""
class ShowTopics(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        canvas = Canvas(self, height=HEIGHT, width=WIDTH)
        canvas.pack()

        background_frame = Frame(self, bg="#7ECFEC")
        background_frame.place(relwidth=1, relheight=1)

        upper_frame = Frame(self, bg="Black")
        upper_frame.place(relx=0.5, rely=0.02, relwidth=0.9, relheight=0.07, anchor='n')
        welcomelabel = Label(upper_frame, text="Topics", fg="Black", font=("Modern", 15, "bold"))
        welcomelabel.place(relwidth=1, relheight=1)

        middle_frame = Frame(self, bg='White')
        middle_frame.place(relx=0.5, rely=0.1, relwidth=0.9, relheight=0.8, anchor='n')
        textlabel = Label(middle_frame, text="You can click here to view your topics, \nthis will open the visualization"
                                             " in your browser. \n If this does not work, please check your version "
                                             "of PyLDAvis.", bg='white', font=('Modern', 15))
        textlabel.place(relwidth=1, relheight=0.8)
        button = Button(middle_frame, text="Show Topics", bg='#7ECFEC', fg='black', borderless = 0, font = 40,
                        command = lambda: visualize())
        button.place(relx=0.3, rely =0.6, relwidth=0.4, relheight=0.3)
        lower_frame = Frame(self, bg='Black', bd=5)
        lower_frame.place(relx=0.5, rely=0.92, relwidth=0.5, relheight=0.07, anchor='n')
        button1 = Button(lower_frame, text='Go Back', bg='white', fg='black', borderless=0, font=40,
                         command=lambda: controller.show_frame("SelectParams"))
        button1.place(relx=0, relwidth=0.5, relheight=1)
        button2 = Button(lower_frame, text='Start all over?', bg='white', fg='black', borderless=0, font=40,
                         command=lambda: controller.show_frame("End"))
        button2.place(relx=0.5, relwidth=0.5, relheight=1)

"""
Allows to start all over with a new corpus by deleting the old one
"""
class End(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        canvas = Canvas(self, height=HEIGHT, width=WIDTH)
        canvas.pack()

        background_frame = Frame(self, bg="#7ECFEC")
        background_frame.place(relwidth=1, relheight=1)

        upper_frame = Frame(self, bg="Black")
        upper_frame.place(relx=0.5, rely=0.02, relwidth=0.9, relheight=0.07, anchor='n')
        welcomelabel = Label(upper_frame, text="Start All Over?", fg="Black", font=("Modern", 15, "bold"))
        welcomelabel.place(relwidth=1, relheight=1)

        middle_frame = Frame(self, bg='White')
        middle_frame.place(relx=0.5, rely=0.1, relwidth=0.9, relheight=0.8, anchor='n')

        textlabel = Label(middle_frame, text="Do you want to use a new corpus?\n For this you "
                                             "have to delete your saved data. \nYou will then have to select a new corpus "
                                             "and \ndo preprocessing again.", bg='white',
                          font=('Modern', 15))
        textlabel.place(rely=0.08, relwidth=1, relheight=0.8)
        textlabel2 = Label(middle_frame, text="Do you want to try new parameters?", bg='white',
                           font=('Modern', 15))
        textlabel2.place(rely=0.05, relwidth=1, relheight=0.15)
        buttonparams = Button(middle_frame, text='Try different parameters', bg='white', fg='black', borderless=0, font=40,
                         command=lambda: controller.show_frame("SelectParams"))
        buttonparams.place(relx=0.3, rely=0.2, relwidth=0.4, relheight=0.1)
        button = Button(middle_frame, text="Delete saved data", bg='#ff4d4d', fg='black', borderless=0, font=40,
                        command=lambda: delete())
        button.place(relx=0.35, rely=0.62, relwidth=0.3, relheight=0.2)
        buttoncorpus= Button(middle_frame, text='Choose New Corpus', bg='white', fg='black', borderless=0, font=40,
                         command=lambda: controller.show_frame("ChooseCorpus"))
        buttoncorpus.place(relx=0.3, rely =0.85, relwidth=0.4, relheight=0.1)


        lower_frame = Frame(self, bg='Black', bd=5)
        lower_frame.place(relx=0.5, rely=0.92, relwidth=0.5, relheight=0.07, anchor='n')
        button1 = Button(lower_frame, text='Go Back', bg='white', fg='black', borderless=0, font=40,
                         command=lambda: controller.show_frame("ShowTopics"))
        button1.place(relx=0, relwidth=0.5, relheight=1)
        button2 = Button(lower_frame, text='Quit Program', bg='white', fg='black', borderless=0, font=40,
                         command=lambda: quit())
        button2.place(relx=0.5, relwidth=0.5, relheight=1)


if __name__ == "__main__":
    app = OrganiseFrames()
    app.mainloop()



    with open('save.txt', 'w') as f:
        for file in files:
            f.write(file + ',')

