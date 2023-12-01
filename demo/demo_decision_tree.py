import pandas as pd
import tkinter as tk
from tkinter.filedialog import askopenfile
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# variables for tkinter
row = 1
border_width = 0.5
width = 17
font = "Consolas"

# variables for decision tree
file_path = None
type_of_tree = None
max_depth = None
min_sample = 2
columns_to_remove = []
label_column = None
test_size = None

main = tk.Tk()
main.title("Demo: Decision Tree")
main.geometry("800x450+100+100")

def modify_string(string):
    width_span = 18
    if len(string) >= width_span:
        return string[:width_span] + "*"
    else:
        return string + " " * (width_span - len(string)) + "*"
      
def get_type_of_tree():
    global type_of_tree
    
    selected_option = MENU_choose_type_of_tree.menu_var.get()
    
    MENU_choose_type_of_tree.config(text=selected_option)
    
    type_of_tree = selected_option
    
    print("type_of_tree = ", type_of_tree, sep="")
    
def get_file_path():
    global file_path
    global label_column
    
    f = askopenfile()
    
    if f is not None:
        file_path = f.name
    else:
        file_path = "No file chosen"
        
    file_name = file_path[file_path.rfind("/")+1:]
    if len(file_name) > 21:
        file_name = "..." + file_name[-26:]
    LABEL_show_file_path.config(text=file_name)
    
    MENU_remove_columns.menu.delete(0, tk.END)
    MENU_choose_label_column.menu.delete(0, tk.END)
    columns_to_remove.clear()
    label_column = None

    if file_path is not None and file_path != "No file chosen" and file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        columns = df.columns.tolist()
        for column in columns:
            var = tk.IntVar()
            MENU_remove_columns.menu.add_checkbutton(label=column,
                                                    variable=var,
                                                    onvalue=1,
                                                    offvalue=0,
                                                    command=lambda v=var, col=column : update_columns_to_remove(v.get(), col))
            MENU_choose_label_column.menu.add_radiobutton(label=column,
                                                          variable=MENU_choose_label_column.menu_var,
                                                          value=column,
                                                          command=get_label_column)
    print("file_path = ", file_path, sep="")
    
def update_columns_to_remove(state, column_name):
    if state == 1:
        columns_to_remove.append(column_name)
    else:
        if column_name in columns_to_remove:
            columns_to_remove.remove(column_name)
    print(columns_to_remove)
        
def show_file_content():
    if file_path and file_path.endswith(".csv"):
        df = pd.read_csv(file_path)

        content_window = tk.Toplevel(main)
        content_window.title("File Content")
        content_window.geometry("900x350+50+200")

        text = tk.Text(content_window, height=25, width=90, font=(font, 12))
        text.pack(padx=20, pady=20)
        text.insert(tk.END, str(df))

        text.config(state="disabled")
    else:
        tk.messagebox.showwarning("No File", "No file selected or file is not in .csv format.")
    
def get_label_column():
    global label_column
    
    selected_option = MENU_choose_label_column.menu_var.get()
    
    MENU_choose_label_column.config(text=selected_option)
    
    if selected_option == "None":
        label_column = None
    else:
        label_column = str(selected_option)
    
    print("label_column = ", label_column, sep="")
    
def alter_entry_max_depth():
    selected_option = MENU_choose_max_depth.menu_var.get()
    
    if selected_option == "None":
        ENTRY_enter_max_depth.delete(0, tk.END)
        ENTRY_enter_max_depth.config(state="disabled")
        MENU_choose_max_depth.config(text="None")
    else:
        ENTRY_enter_max_depth.config(state="normal")
        ENTRY_enter_max_depth.config(bg="#ABEBC6")
        MENU_choose_max_depth.config(text="Other")

def alter_entry_min_sample():
    selected_option = MENU_choose_min_sample.menu_var.get()
    
    if selected_option == "2":
        ENTRY_enter_min_sample.delete(0, tk.END)
        ENTRY_enter_min_sample.config(state="disabled")
        MENU_choose_min_sample.config(text="2")
        is_min_sample_chosen = True
    else:
        ENTRY_enter_min_sample.config(state="normal")
        ENTRY_enter_min_sample.config(bg="#ABEBC6")
        MENU_choose_min_sample.config(text="Other")
        
def get_train_test_split_ratio():
    global test_size
    
    selected_option = MENU_choose_train_test_split.menu_var.get()
    
    if selected_option == "0.0":
        MENU_choose_train_test_split.config(text="100% train, 0% test")
    else:
        MENU_choose_train_test_split.config(text="70% train, 30% test")
    
    test_size = float(selected_option)
    
    print("test_size = ", test_size, sep="")
    print(type(test_size))
    
def build_classification_tree(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(max_depth=max_depth,
                                 min_samples_split=min_sample, 
                                 random_state=2)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    accuracy_str = f"Accuracy: {accuracy:.2f}"
    LABEL_accuracy.config(text=accuracy_str)
    
    plt.figure(figsize=(12, 12))
    tree.plot_tree(clf,
                    feature_names=X_train.columns.tolist(),
                    class_names=y_train.unique().tolist(),
                    filled=True,
                    rounded=True,
                    fontsize=7,
                    max_depth=None)

    plt.show()
    
def build_regression_tree(X_train, y_train, X_test, y_test):
    reg = DecisionTreeRegressor(max_depth=max_depth,
                                min_samples_split=min_sample, 
                                random_state=2)
    reg.fit(X_train, y_train)
    
    y_pred = reg.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    
    r2_str = f"R2 score: {r2:.2f}"
    LABEL_accuracy.config(text=r2_str)

    plt.figure(figsize=(12, 12))
    tree.plot_tree(reg,
                    feature_names=X_train.columns.tolist(),
                    class_names=y_train.unique().tolist(),
                    filled=True,
                    rounded=True,
                    fontsize=7,
                    max_depth=None)

    plt.show()

def submit():
    global max_depth
    global min_sample
    global test_size
    
    if ENTRY_enter_max_depth["state"] == "normal":
        max_depth = int(ENTRY_enter_max_depth.get())
    else:
        max_depth = None
        
    if ENTRY_enter_min_sample["state"] == "normal":
        min_sample = int(ENTRY_enter_min_sample.get())
    else:
        min_sample = 2

    print("\n-------SUBMIT-------")
    print("file_path =", file_path)
    print("type_of_tree =", type_of_tree)
    print("max_depth =", max_depth)
    print("min_sample =", min_sample)
    print("columns_to_remove =", columns_to_remove)
    print("label_column =", label_column)
    print("test_size =", test_size)
    
    df = pd.read_csv(file_path)
        
    X = df.drop(columns_to_remove+[label_column], axis=1)
    y = df[label_column]
        
    if test_size == 0.0: # To ensure that the tree remains the same each time the program runs
        X_train = X 
        y_train = y
        X_test = X
        y_test = y
    else:
        (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=test_size, random_state=2)
        
    if type_of_tree == "Classification":
        y_train = y_train.astype("str")
        y_test = y_test.astype("str")
        build_classification_tree(X_train, y_train, X_test, y_test)
    else:
        build_regression_tree(X_train, y_train, X_test, y_test)
        
LABEL_file_path = tk.Label(main,
                           text=modify_string("Choose file:"),
                           font=(font, 20))
LABEL_file_path.grid(row=row, column=1,ipady=5)
LABEL_show_file_path = tk.Label(main, 
                           text=file_path, 
                           font=(font, 20),
                           anchor="w",
                           justify="right",
                           width=21,
                           relief="solid",
                           borderwidth=border_width)
LABEL_show_file_path.grid(row=row, column=2, columnspan=2)
BUTTON_browse_file = tk.Button(main, 
                        text="Browse", 
                        font=(font, 15),
                        command=get_file_path)
BUTTON_browse_file.grid(row=row, column=4)
BUTTON_show_file = tk.Button(main, 
                        text="Show", 
                        font=(font, 15),
                        command=show_file_content)
BUTTON_show_file.grid(row=row, column=5)
row += 1

LABEL_type_of_tree = tk.Label(main, 
                              text=modify_string("Type of tree:"), 
                              font=(font, 20))
LABEL_type_of_tree.grid(row=row, column=1)
MENU_choose_type_of_tree = tk.Menubutton(main, 
                                         text="Choose", 
                                         font=(font, 20),
                                         width=width,
                                         relief="solid",
                                         borderwidth=border_width)
MENU_choose_type_of_tree.menu = tk.Menu(MENU_choose_type_of_tree, tearoff=0)
MENU_choose_type_of_tree.menu_var = tk.StringVar()
MENU_choose_type_of_tree.menu_var.set("")
MENU_choose_type_of_tree["menu"] = MENU_choose_type_of_tree.menu
MENU_choose_type_of_tree.menu.add_radiobutton(label="Classification",
                                              variable=MENU_choose_type_of_tree.menu_var,
                                              value="Classification",
                                              command=get_type_of_tree)
MENU_choose_type_of_tree.menu.add_radiobutton(label="Regression", 
                                              variable=MENU_choose_type_of_tree.menu_var, 
                                              value="Regression", 
                                              command=get_type_of_tree)
MENU_choose_type_of_tree.grid(row=row, column=2)
row += 1
    
LABEL_max_depth = tk.Label(main, 
                           text=modify_string("Max depth:"), 
                           font=(font, 20))
LABEL_max_depth.grid(row=row, column=1, ipady=5)
MENU_choose_max_depth = tk.Menubutton(main, 
                                      text="Choose", 
                                      font=(font, 20),
                                      width=width,
                                      relief="solid",
                                      borderwidth=border_width)
MENU_choose_max_depth.menu = tk.Menu(MENU_choose_max_depth, tearoff=0)
MENU_choose_max_depth["menu"] = MENU_choose_max_depth.menu
MENU_choose_max_depth.menu_var = tk.StringVar()
MENU_choose_max_depth.menu_var.set("")
MENU_choose_max_depth.menu.add_radiobutton(label="None",
                                           variable=MENU_choose_max_depth.menu_var,
                                           value="None",
                                           command=alter_entry_max_depth)
MENU_choose_max_depth.menu.add_radiobutton(label="Other",
                                           variable=MENU_choose_max_depth.menu_var,
                                           value="Other",
                                           command=alter_entry_max_depth)
MENU_choose_max_depth.grid(row=row, column=2)
ENTRY_enter_max_depth = tk.Entry(main,
                                 font=(font, 20),
                                 state="disabled",
                                 width=4)
ENTRY_enter_max_depth.grid(row=row, column=3)                               
row += 1
        
LABEL_min_sample = tk.Label(main, 
                            text=modify_string("Min sample:"), 
                            font=(font, 20))
LABEL_min_sample.grid(row=row, column=1)
MENU_choose_min_sample = tk.Menubutton(main, 
                                       text="Choose", 
                                       font=(font, 20),
                                       width=width,
                                       relief="solid",
                                       borderwidth=border_width)
MENU_choose_min_sample.menu = tk.Menu(MENU_choose_min_sample, tearoff=0)
MENU_choose_min_sample["menu"] = MENU_choose_min_sample.menu
MENU_choose_min_sample.menu_var = tk.StringVar()
MENU_choose_min_sample.menu_var.set("")
MENU_choose_min_sample.menu.add_radiobutton(label="2",
                                           variable=MENU_choose_min_sample.menu_var,
                                           value="2",
                                           command=alter_entry_min_sample)
MENU_choose_min_sample.menu.add_radiobutton(label="Other",
                                           variable=MENU_choose_min_sample.menu_var,
                                           value="Other",
                                           command=alter_entry_min_sample)
MENU_choose_min_sample.grid(row=row, column=2)
ENTRY_enter_min_sample = tk.Entry(main,
                                 font=(font, 20),
                                 state="disabled",
                                 width=4)
ENTRY_enter_min_sample.grid(row=row, column=3)       
row += 1

LABEL_remove_columns = tk.Label(main,
                                text="Remove columns:", 
                                font=(font, 20))
LABEL_remove_columns.grid(row=row, column=1, sticky="w")
MENU_remove_columns = tk.Menubutton(main,
                                    text="Choose",
                                    font=(font, 20),
                                    width=width,
                                    relief="solid",
                                    borderwidth=border_width)
MENU_remove_columns.menu = tk.Menu(MENU_remove_columns, tearoff=0)
MENU_remove_columns["menu"] = MENU_remove_columns.menu
MENU_remove_columns.grid(row=row, column=2)
row += 1

LABEL_label_column = tk.Label(main,
                              text=modify_string("Target column:"),
                              font=(font, 20))
LABEL_label_column.grid(row=row, column=1)
MENU_choose_label_column = tk.Menubutton(main,
                                        text="Choose",
                                        font=(font, 20),
                                        width=width,
                                        relief="solid",
                                        borderwidth=border_width)
MENU_choose_label_column.menu = tk.Menu(MENU_choose_label_column, tearoff=0)
MENU_choose_label_column["menu"] = MENU_choose_label_column.menu
MENU_choose_label_column.menu_var = tk.StringVar()
MENU_choose_label_column.menu_var.set("")
MENU_choose_label_column.grid(row=row, column=2)
row += 1

LABEL_train_test = tk.Label(main, 
                            text=modify_string("Train-Test split:"), 
                            font=(font, 20))
LABEL_train_test.grid(row=row, column=1)
MENU_choose_train_test_split = tk.Menubutton(main,
                                             text="Choose", 
                                             font=(font, 20),
                                             relief="solid",
                                             width=19,
                                             borderwidth=border_width)
MENU_choose_train_test_split.menu = tk.Menu(MENU_choose_train_test_split, tearoff=0)
MENU_choose_train_test_split["menu"] = MENU_choose_train_test_split.menu
MENU_choose_train_test_split.menu_var = tk.StringVar()
MENU_choose_train_test_split.menu_var.set("")
MENU_choose_train_test_split.menu.add_radiobutton(label="100% train, 0% test",
                                                  variable=MENU_choose_train_test_split.menu_var,
                                                  value=0.0,
                                                  command=get_train_test_split_ratio)
MENU_choose_train_test_split.menu.add_radiobutton(label="70% train, 30% test",
                                                  variable=MENU_choose_train_test_split.menu_var,
                                                  value=0.3,
                                                  command=get_train_test_split_ratio)
MENU_choose_train_test_split.grid(row=row, column=2)    
row += 1

BUTTON_submit = tk.Button(main, 
                          text="Submit", 
                          font=(font, 20),
                          command=submit)
BUTTON_submit.grid(row=row, column=2, pady=10)
row += 1

LABEL_accuracy = tk.Label(main,
                          text="",
                          font=(font, 20))
LABEL_accuracy.grid(row=row, column=1,ipady=5)
row += 1

main.mainloop()