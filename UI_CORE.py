#Author: Cosmos Bennett
#Cosmos's Ai Neural Net Fraud Analysis Tool v0.1
#Uses neural network to do credit card fraud detection



############
#Cosmos's installation notes

#1. get python 3.63

#2. python -m pip install PyQt5


print("Loading Ai Module ...")

import cosmos_ai_credit_card_fraud_detection_load_pretrained as AI_NEURAL_AGENT

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QToolButton, QTextEdit
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt,QSize,QPoint

from PyQt5.QtGui import QTextDocument, QTextCursor, QColor, QSyntaxHighlighter, QTextCharFormat
from PyQt5.QtCore import QRegExp

import re
import asyncio

import tkinter as tk
from tkinter import filedialog


class NcbAiFraudAnalyser(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cosmos Ai Neural Fraud Analyser")
        self.setGeometry(200, 200, 800, 850)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.output_text = "";
        
        # Create the title label
        title_label = QLabel("Cosmos Ai Neural Fraud Analyser", self)
        title_label.setFont(QFont('Segoe UI', 40))
        title_label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        title_label.setFixedHeight(70)
        
        # Create the title label
        self.status_label = QLabel("..", self)
        self.status_label.setFont(QFont('Segoe UI', 12))
        self.status_label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.status_label.setFixedHeight(70)

        
        

        # Output field
        self.output_field = QTextEdit(self)
        self.output_field.setReadOnly(True)
        self.output_field.setFixedHeight(550)
        self.output_field.setStyleSheet("""
            QTextEdit {
                background-color: darkGray;
                color: #333;
                font-family: Consolas, Courier, monospace;
                font-size: 12px;
                width: 1600px;
                border: none;
            }
        """)

        #Place holder function when program starts up
        self.output_field_text_test = "Account Prediction\n"
        self.output_field_text_test += "120344  Fraudulent \n"
        self.output_field_text_test += "120344  Non-Fraudulent \n"

        self.file_path = ""
        self.output_string = ""

        #Example: "python code to generate fibonacci output"
        #output_field_highlighter = SyntaxHighlighter(self.output_field.document())
        #self.output_field.setText(output_field_text)
        

        
        # Create the exit button
        exit_button = QToolButton(self)
        exit_button.setIcon(QIcon("data/gui/close.png"))
        exit_button.setIconSize(QSize(30, 30))
        exit_button.setAutoRaise(True)
        exit_button.clicked.connect(self.close)
        exit_button.setFixedSize(40, 40)
        exit_button.move(self.width() - exit_button.width() - 10, 10)
        #set hover color
        exit_button.setStyleSheet("QToolButton { border: none; } QToolButton:hover { background-color: darkGray; }")

        
        # send button
        send_button = QToolButton(self)
        send_button.setIcon(QIcon("data/gui/file.png"))
        send_button.setIconSize(QSize(90, 90))
        send_button.setAutoRaise(True)
        send_button.setFixedSize(90, 90)
        #adjust send button
        new_pos = send_button.pos() + QPoint(200, 130)
        send_button.move(new_pos)
        #set hover color
        send_button.setStyleSheet("QToolButton { border: none; } QToolButton:hover { background-color: darkGray; }")

        # clear button
        clear_button = QToolButton(self)
        clear_button.setIcon(QIcon("data/gui/close.png"))
        clear_button.setIconSize(QSize(90, 90))
        clear_button.setAutoRaise(True)
        clear_button.setFixedSize(90, 90)
        #adjust send button
        new_pos = clear_button.pos() + QPoint(340, 130)
        clear_button.move(new_pos)
        #set hover color
        clear_button.setStyleSheet("QToolButton { border: none; } QToolButton:hover { background-color: darkGray; }")
		
		
        # save button
        save_button = QToolButton(self)
        save_button.setIcon(QIcon("data/gui/send.png"))
        save_button.setIconSize(QSize(90, 90))
        save_button.setAutoRaise(True)
        save_button.setFixedSize(90, 90)
        #adjust send button
        new_pos = save_button.pos() + QPoint(460, 130)
        save_button.move(new_pos)
        #set hover color
        save_button.setStyleSheet("QToolButton { border: none; } QToolButton:hover { background-color: darkGray; }")
		
		

        # Create the vertical layout for the title and input field/button
        main_layout = QVBoxLayout()
        main_layout.addWidget(title_label)
        main_layout.addStretch(1)
        main_layout.addWidget(self.output_field)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Set the main layout for the window
        self.setLayout(main_layout)

        def clear_output():
            self.output_field.setText("")
            self.output_string = "" #reset program data string
            self.status_label.setText("")
            
            
        from PyQt5.QtCore import QTimer



        def ai_output_non_animated():
            # Apply syntax highlighting to the output field
            #highlighter = SyntaxHighlighter(self.output_field.document())


            ###################
            # Generate ai predictions and Ui text
            output = self.output_field_text_test

            
            # Add the response to the output field
            self.output_field.insertPlainText(output)

            #Clear output text variable
            self.output_text = "";


        def ai_output_animated( ):
            
            ###################
            # Generate ai predictions and Ui text
            output = self.output_string


            # animate output
            output_texts_parsed = re.findall(r'\w+|\W+', output) # find both words and symbols
            delay = 60 # milliseconds

            ###################
            #Results for GPT4ALL_J or GPT4ALL models


            # Apply syntax highlighting to the output field
            #highlighter = SyntaxHighlighter(self.output_field.document())   
            for i, token in enumerate(output_texts_parsed):
                if re.match(r'^\W+$', token): # if it's a symbol
                    QTimer.singleShot((i+1)*delay, lambda symbol=token: self.output_field.insertPlainText(symbol))
                else: # if it's a word
                    QTimer.singleShot((i+1)*delay, lambda word=token: self.output_field.insertPlainText(word + ''))         


            
            #Clear output text variable
            self.output_text = "";
        
        
        def highlightDynamicResponse ( ):
            clear_output ( );
            
            # Re-establish output_field with highlighted content
            # Apply syntax highlighting to the output field
            highlighter = SyntaxHighlighter(self.output_field.document())
            
            # Add the response to the output field
            self.output_field.insertPlainText(self.output_text)
            
            # Make the window draggable
            self.dragPos = QPoint()


        def getPrediction (value):
            returnValue = ""
            if value >= 0.07: #for 0.07 is basis threshold
                returnValue = "Fraudulent"
            else:
                returnValue = "Non-Fraudulent"
                
            return returnValue
        

         
        def getCn ( array ):
               return ' '.join(str(x) for x in array)
            
        def process_file():
            last_file_path = self.file_path
            
            space = "          "
            
            self.output_string = "File Analysed : " + last_file_path + "\n\n"

            self.output_string += "Transaction Acc" + space + "Prediction" + space + "Probability\n\n"
            
            
            if last_file_path:
                data=AI_NEURAL_AGENT.PROCESS_DATA(last_file_path)
                
               
                for d in data:
                    rawPrediction = AI_NEURAL_AGENT.doOnlineInferenceOnRawRecord(getCn(d))
                    #print("pred " + str(rawPrediction))
                    self.output_string += str(d[0]) + space + getPrediction(rawPrediction)  + space + str(rawPrediction)  + "\n" 
                
                
            #self.output_field.insertPlainText(self.output_string)
            ai_output_animated ( )
            self.status_label.setText("done!")
           
        def load_file():
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            self.status_label.setText("processing...")
            self.file_path = filedialog.askopenfilename()
            process_file()

        def save_predictions():
            # Define the file path where you want to save the string
            out_path = "predictions.txt"

            # Open the file in write mode ('w' mode)
            with open(out_path, "w") as file:
                # Write the string to the file
                file.write(self.output_string)

            self.status_label.setText("saved!")

            
        send_button.clicked.connect(load_file)
        clear_button.clicked.connect(clear_output)
        save_button.clicked.connect(save_predictions)
        
        # Make the window draggable
        self.dragPos = QPoint( )
            
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragPos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.dragPos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self.dragPos = None




        
        
if __name__ == '__main__':
    app = QApplication([])
    ncbfinchat_ui = NcbAiFraudAnalyser()
    ncbfinchat_ui.show()
    app.exec_()
