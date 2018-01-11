from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from PIL import Image
import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()


def initialization():
    w = 650  # width for the Tk root
    h = 550  # height for the Tk root

    # get screen width and height
    ws = root.winfo_screenwidth()  # width of the screen
    hs = root.winfo_screenheight()  # height of the screen

    # calculate x and y coordinates for the Tk root window
    x = (ws / 2) - (w / 2)
    y = (hs / 2) - (h / 2)

    # set the dimensions of the screen
    # and where it is placed
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))

    # *** Browse Button ***
    browse_button = Button(frame_train, text='Browse', width=5, fg='blue', bg='white', command=take_image)
    browse_button.config(font=('Ubuntu', 12))
    browse_button.place(x=280, y=285)

    # *** Train Button ***
    train_button = Button(frame_train, text='Train', width='5', fg='blue', bg='white', command=training_area)
    train_button.config(font=('Ubuntu', 12))
    train_button.place(x=280, y=95)

    # *** Predict Button ***
    predict_button = Button(frame_predict, text='Predict', width='5', fg='blue', bg='white', command=predicting_area)
    predict_button.config(font=('Ubuntu', 12))
    predict_button.place(x=400, y=40)

    # *** Header of Training Frame's Label ***
    start_label = Label(frame_train, text='Training Area', fg='brown')
    start_label.config(font=('Ubuntu', 16))
    start_label.place(x=250, y=0)

    img = PhotoImage(file="Icons/training.png")
    photo_label = Label(frame_train, image=img)
    photo_label.grid()
    photo_label.image = img
    photo_label.config(height=100, width=100)
    photo_label.place(x=0, y=0)

    # *** Header of Prediction Frame's Label ***
    start_label1 = Label(frame_predict, text='Prediction Area', fg='brown')
    start_label1.config(font=('Ubuntu', 16))
    start_label1.place(x=245, y=0)

    img = PhotoImage(file="Icons/predicting.png")
    photo_label = Label(frame_predict, image=img)
    photo_label.grid()
    photo_label.image = img
    photo_label.config(height=100, width=100)
    photo_label.place(x=550, y=0)

    # *** Labels for All Buttons ***
    label = Label(frame_train, text='Press the Button to Train the Data', bg='gray')
    label.config(font=('Ubuntu', 12))
    label.place(x=180, y=65)

    img = PhotoImage(file="Icons/trainClick.png")
    photo_label = Label(frame_train, image=img)
    photo_label.grid()
    photo_label.image = img
    photo_label.config(height=40, width=40)
    photo_label.place(x=295, y=130)

    img = PhotoImage(file="Icons/trainClick.png")
    photo_label = Label(frame_train, image=img)
    photo_label.grid()
    photo_label.image = img
    photo_label.config(height=40, width=40)
    photo_label.place(x=295, y=320)

    browse_label = Label(frame_train, text='Please Browse the Picture', bg='gray')
    browse_label.config(font=('Ubuntu', 12))
    browse_label.place(x=75, y=290)

    label2 = Label(frame_predict, text='Press the Button to See the Predicted Digit', bg='white')
    label2.config(font=('Ubuntu', 12))
    label2.place(x=65, y=45)

    img = PhotoImage(file="Icons/predictClick.png")
    photo_label = Label(frame_predict, image=img)
    photo_label.grid()
    photo_label.image = img
    photo_label.config(height=40, width=40)
    photo_label.place(x=420, y=75)

    # *** Credit Button ***
    credit_button = Button(frame_predict, text='Credit', width='5', fg='blue', bg='white', command=credit_details)
    credit_button.config(font=('Ubuntu', 12))
    credit_button.place(x=0, y=135)

    # *** About Button ***
    about_button = Button(frame_train, text='About', width='5', fg='blue', bg='white', command=project_details)
    about_button.config(font=('Ubuntu', 12))
    about_button.place(x=576, y=345)


def credit_details():
    messagebox.showinfo("Created By", "Shahnawaz Hossan\n2012331531\nDept. of CSE\nSylhet Engineering College")


def project_details():
    messagebox.showinfo("Project's Details",
                        "It's a machine learning project which is trained by "
                        "42000 handwritten digits.Every data has 784 gray scale "
                        "value of a 28x28 image of a handwritten digit(0 to 9)."
                        "It takes a 28x28 image of a handwritten digit and can "
                        "predict which digit is this.")


def take_image():
    filename = filedialog.askopenfilename(initialdir="/home/pabon/Desktop", title="Select file",
                                          filetypes=(("png files", "*.png"), ("All Files", "*.*")))

    img = PhotoImage(file=filename)

    # *** Image Label ***
    photo_label = Label(frame_train, image=img)
    photo_label.grid()
    photo_label.image = img
    photo_label.config(height=75, width=75)
    photo_label.place(x=370, y=260)

    # *** Load Image and Convert it into Pixel intensity values ***
    im = Image.open(filename).convert('L')
    data = np.array(im)  # Store gray scale values of each pixel

    f = open('/home/pabon/Downloads/store.csv', 'w')
    string = ""
    for i in range(28 * 28):
        s = str(i)
        f.write("pixel")
        f.write(s)
        if i < 783:
            f.write(",")

    f.write("\n")

    counter = 0

    for i in range(28):
        for j in range(28):
            string = string + str(data[i][j])
            if counter < 783:
                string = string + ","
                counter = counter + 1

    string = string + '\n'
    # print('String :', string)
    f.write(string)
    f.close()


def training_area():
    data = pd.read_csv("/home/pabon/Downloads/train.csv").as_matrix()
    x_train = data[0:42000, 1:]
    train_label = data[0:42000, 0]
    clf.fit(x_train, train_label)

    img = PhotoImage(file="Icons/tick.png")
    photo_label = Label(frame_train, image=img)
    photo_label.grid()
    photo_label.image = img
    photo_label.config(height=40, width=40)
    photo_label.place(x=90, y=170)

    # *** Showing Label ***
    show_label = Label(frame_train, text='Data Training has been completed Successfully', bg='gray')
    show_label.config(font=('Ubuntu', 14))
    show_label.place(x=140, y=180)


def predicting_area():
    test_data = pd.read_csv("/home/pabon/Downloads/store.csv").as_matrix()
    temp = test_data[0:, 0:]
    predict_input = clf.predict([temp[0]])
    string_of_predict_input = "Predicted Handwritten Digit is : "
    string_of_predict_input += str(predict_input[0])

    # *** Showing Prediction ***
    predict_label = Label(frame_predict, text=string_of_predict_input, bg='white', fg='red')
    predict_label.config(font=('Ubuntu', 14))
    predict_label.place(x=300, y=130)


def predict_from_test_file():
    test_data = pd.read_csv("/home/pabon/Downloads/test.csv").as_matrix()
    test_data = test_data[0:, 0:]
    # print('Type of test_data :', type(test_data[1050]))
    # print('Shape:', test_data[1050].shape)
    print('test_data[1050]', test_data[1050])
    x_test = test_data[1050]
    x_test.shape = (28, 28)
    pt.imshow(255 - x_test, cmap='gray')
    print('Predicting Output is : ', clf.predict([test_data[1050]]))
    pt.show()


if __name__ == '__main__':
    root = Tk()
    root.title('Handwritten Digit Recognition')  # Set Title of the Frame

    frame_train = Frame(root, bg='gray', width=650, height=380)
    frame_train.pack(side=TOP)

    frame_predict = Frame(root, bg='white', width=650, height=270)
    frame_predict.pack(side=BOTTOM)

    initialization()

root.mainloop()
