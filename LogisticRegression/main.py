from tkinter import *
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np


def init():
    # *** Show Label ***
    show_label = Label(frame, text='Press the Show Button to see the Training Data')
    show_label.config(font=('Times New Roman', 14))
    show_label.place(x=120, y=20)

    # *** Show Button ***
    show_button = Button(frame, text='Show', width=5, fg='brown', bg='white', command=plot_data)
    show_button.config(font=('Times New Roman', 12))
    show_button.place(x=270, y=50)

    # *** Decision Boundary Label ***
    label = Label(frame, text='Press the Decision Boundary Button to see the Decision Boundary')
    label.config(font=('Times New Roman', 14))
    label.place(x=50, y=130)

    # *** Show Decision Boundary Button ***
    decision_button = Button(frame, text='Decision Boundary', width=15, fg='brown', bg='white', command=training)
    decision_button.config(font=('Times New Roman', 12))
    decision_button.place(x=230, y=160)

    # *** Predict Label 1 ***
    predict_label1 = Label(frame, text='Enter Score 1:')
    predict_label1.config(font=('Times New Roman', 14))
    predict_label1.place(x=100, y=300)

    # *** Predict Label 2 ***
    predict_label2 = Label(frame, text='Enter Score 2:')
    predict_label2.config(font=('Times New Roman', 14))
    predict_label2.place(x=100, y=350)

    # *** Predict Button ***
    decision_button = Button(frame, text='Predict', width=5, fg='brown', bg='white', command=prediction)
    decision_button.config(font=('Times New Roman', 12))
    decision_button.place(x=250, y=390)


def plot_data():
    plt.scatter(pos[:, 0], pos[:, 1], c='blue', marker='+', label="Admitted")
    plt.scatter(neg[:, 0], neg[:, 1], c='red', marker='^', label="Not Admitted")
    plt.legend(loc='best')
    plt.title('Mark Details')
    plt.xlabel('Mark 1')
    plt.ylabel('Mark 2')
    plt.show()


def prediction():
    a = float(score1.get("1.0", "end-1c"))
    b = float(score2.get("1.0", "end-1c"))

    test_data = [a, b]

    print(test_data)

    prdct = clf.predict([test_data])

    print(prdct)

    accepted = "The Student will be Accepted"
    denied = "The Student will be Rejected"
    accept = Label(frame, text=accepted, fg='green')
    accept.config(font=('Times New Roman', 14))
    deny = Label(frame, text=denied, fg='red')
    deny.config(font=('Times New Roman', 14))

    if prdct[0] == 1:
        deny.place_forget()
        accept.place(x=180, y=450)
    else:
        accept.place_forget()
        deny.place(x=180, y=450)


def training():
    log_reg = clf.fit(x_train, train_label)

    # *** Plot the data with decision boundary *** #
    coef = log_reg.coef_
    intercept = log_reg.intercept_

    # see the coutour approach for a more general solution
    ex1 = np.linspace(30, 100, 100)
    ex2 = -(coef[:, 0] * ex1 + intercept[:, 0]) / coef[:, 1]

    plt.scatter(pos[:, 0], pos[:, 1], c='blue', marker='+', label="Admitted")
    plt.scatter(neg[:, 0], neg[:, 1], c='red', marker='^', label="Not Admitted")
    plt.plot(ex1, ex2, color='black', label='decision boundary')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    root = Tk()
    root.title('Admitted or Not Admitted')  # Set Title of the Frame

    frame = Frame(root, width=650, height=500)
    frame.pack()

    # *** Data Retrieve and Label Set *** #
    data = pd.read_csv(
        "/home/pabon/Dropbox/PythonPractice/Machine Learning in Python/LogisticRegression/data.csv").as_matrix()

    target = data[0:100, 2]
    pos = data[target == 1]
    neg = data[target == 0]
    x_train = data[0:100, 0:2]
    train_label = data[0:100, 2]

    clf = OneVsRestClassifier(LogisticRegression(penalty='l1'))

    # *** Text Fields *** #
    score1 = Text(frame, height=1, width=15, fg='blue')
    score2 = Text(frame, height=1, width=15, fg='blue')
    score1.place(x=230, y=300)
    score2.place(x=230, y=350)
    score1.config(font=('Times New Roman', 14))
    score2.config(font=('Times New Roman', 14))

    init()

root.mainloop()
