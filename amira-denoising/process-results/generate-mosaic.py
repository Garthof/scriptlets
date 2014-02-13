#!/usr/bin/env python

from Tkinter import *

class App:
    def __init__(self, master):
        self.lblDir = Label(master, text="Directory")
        self.lblDir.grid(row=0, column=0)

        self.txtDir = Entry(master, width=60)
        self.txtDir.grid(row=0, column=1, columnspan=2)

        self.varIter = IntVar();
        self.btnIter = Checkbutton(master, text="Iter", variable=self.varIter)
        self.btnIter.grid(row=1, column=0, sticky=W)

        self.txtIter = Entry(master)
        self.txtIter.grid(row=1, column=1, sticky=W)

        self.varPatch = IntVar();
        self.btnPatch = Checkbutton(master, text="Patch", variable=self.varPatch)
        self.btnPatch.grid(row=2, column=0, sticky=W)

        self.txtPatch = Entry(master)
        self.txtPatch.grid(row=2, column=1, sticky=W)

        self.varPCA = IntVar();
        self.btnPCA = Checkbutton(master, text="PCA", variable=self.varPCA)
        self.btnPCA.grid(row=3, column=0, sticky=W)

        self.txtPCA = Entry(master)
        self.txtPCA.grid(row=3, column=1, sticky=W)

        self.varStdDev1 = IntVar();
        self.btnStdDev1 = Checkbutton(master, text="StdDev1", variable=self.varStdDev1)
        self.btnStdDev1.grid(row=4, column=0, sticky=W)

        self.txtStdDev1 = Entry(master)
        self.txtStdDev1.grid(row=4, column=1, sticky=W)

        self.varStdDev2 = IntVar();
        self.btnStdDev2 = Checkbutton(master, text="StdDev2", variable=self.varStdDev2)
        self.btnStdDev2.grid(row=5, column=0, sticky=W)

        self.txtStdDev2 = Entry(master)
        self.txtStdDev2.grid(row=5, column=1, sticky=W)

        self.lblImg = Label(master, text="Test")
        self.lblImg.grid(row=1, column=2, rowspan=5, sticky=W+E+N+S)

    def say_hi(self):
        print "hi there, everyone!"


def main():
    root = Tk()
    app = App(root)

    root.mainloop()

if __name__ == "__main__":
    main()

