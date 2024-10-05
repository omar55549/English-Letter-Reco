import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model


class LetterRecognizerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Letter Recognition")
        self.master.geometry("600x600")  # Set a larger window size

        # Set the background color of the window to white
        self.master.configure(bg='white')

        # Create a frame around the canvas with a thick border
        canvas_frame = tk.Frame(master, bg='black', bd=5)  # Border color is black, border width is 5
        canvas_frame.pack(pady=10)

        # Increase the size of the canvas (e.g., 500x500) and place it inside the frame
        self.canvas = tk.Canvas(canvas_frame, width=500, height=500, bg='white')
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.canvas.bind("<Motion>", self.update_marker)

        # Adjust the image size to match the new canvas size
        self.image = Image.new("L", (500, 500), 0)  # Create a black image of the same size as the canvas
        self.draw = ImageDraw.Draw(self.image)

        # Load the model once
        self.model = load_model('Model_Weights/letter_recognition_cnn_st.h5')

        # Create a frame for buttons
        button_frame = tk.Frame(master, bg='white')
        button_frame.pack(pady=10)

        # Create a button for recognizing letters
        self.button_recognize = tk.Button(button_frame, text="Recognize Letter", command=self.recognize_letter, font=("Arial", 18), bg='green', fg='white')
        self.button_recognize.pack(side=tk.LEFT, padx=5)

        # Create a button for clearing the canvas
        self.button_clear = tk.Button(button_frame, text="Clear", command=self.clear_canvas, font=("Arial", 18), bg='green', fg='white')
        self.button_clear.pack(side=tk.LEFT, padx=5)

        # Label to display the predicted letter
        self.label_result = tk.Label(master, text="", font=("Arial", 20), bg='white', fg='black')
        self.label_result.pack(pady=10)

        # Marker setup
        self.marker = None

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill='black', outline='black')  # Change the marker to black on the white background
        self.draw.rectangle([x-5, y-5, x+5, y+5], fill='white')

    def update_marker(self, event):
        # Remove the previous marker
        if self.marker:
            self.canvas.delete(self.marker)

        # Draw a new marker at the current mouse position
        self.marker = self.canvas.create_oval(event.x-5, event.y-5, event.x+5, event.y+5, fill='black', outline='black')

    def release(self, event):
        pass

    def img_preprocess(self):
        # Resize the image to 64x64 pixels (even though the canvas is larger)
        resized_image = self.image.resize((64, 64), Image.LANCZOS)

        img_array = np.array(resized_image) / 255.0  # Normalize

        # Reshape to (1, 64, 64, 1)
        img_array = img_array.reshape(1, 64, 64, 1)  # Add batch and channel dimensions

        return img_array

    def recognize_letter(self):
        img_array = self.img_preprocess()

        # Make predictions
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions)

        # Class labels
        class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        predicted_letter = class_labels[predicted_class]

        # Update the label with the predicted letter
        self.label_result.config(text=f"Predicted Letter: {predicted_letter}")

    def clear_canvas(self):
        # Clear the canvas and reset the image
        self.canvas.delete("all")
        self.image = Image.new("L", (500, 500), 0)  # Reset to a black image of the new size
        self.draw = ImageDraw.Draw(self.image)
        self.marker = None  # Reset the marker

        # Clear the result label
        self.label_result.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    app = LetterRecognizerApp(root)
    root.mainloop()
