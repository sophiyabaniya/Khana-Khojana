import os
import secrets
from PIL import Image
from flask import render_template, url_for, flash, redirect, request, abort
from flaskblog import app, db, bcrypt, mail
from flaskblog.forms import (RegistrationForm, LoginForm, UpdateAccountForm,
                             PostForm, RequestResetForm, ResetPasswordForm)
from flaskblog.models import User, Post
from flask_login import login_user, current_user, logout_user, login_required
from flask_mail import Message


import pandas as pd
import numpy as np
import math
import operator

from sklearn.model_selection import train_test_split 

from sklearn.preprocessing import StandardScaler 

import os
import gensim
from gensim import models
from gensim.models import Word2Vec, KeyedVectors
from gensim import corpora

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#### Start of STEP 1
# Importing data 
import csv
with open(r'C:\Users\sophiya\Desktop\Flask_Blog\flaskblog\hamrofinaldata.csv', 'r') as csv_file:
    data = pd.read_csv(csv_file)
#### End of STEP 1

pd.set_option('display.max_colwidth', -1)

@app.route("/")

@app.route("/home")
def home():
    page = request.args.get('page', 1, type=int)
    posts = Post.query.order_by(Post.date_posted.desc()).paginate(page=page, per_page=5)
    return render_template('home.html', posts=posts)

@app.route("/about")
def about():
         return render_template("about.html")

@app.route("/home/dashboard", methods=['GET', 'POST'])
@login_required
def dashboard():

    page=request.args.get('page', 1, type=int)
    posts=Post.query.order_by(Post.date_posted.desc()).paginate(page=page, per_page=5)
    return render_template('dashboard.html', posts=posts)
    
@app.route("/home/recipe")
def recipe():
         
         return render_template("recipe.html")


@app.route("/recipemydisplay")
def recipemydisplay():
         
         return render_template("recipemydisplay.html")


@app.route("/recipemy", methods=["GET", "POST"])
def recipemy():
    bar = request.form['test']
    new_ing = []
    new_ing = bar.split(',')
         
    ingredients2 = ['xyz']
    ingredients = data["Ingredients"].tolist()
    for i in range(1000):
        ingredients1 = ingredients[i].split(',')
        ingredients2.extend(ingredients1)
    ingredients2.remove('xyz')

    mylist = list(dict.fromkeys(ingredients2))
    mylist.sort()

    resultrecipe = []
    for item in mylist:
        if mylist.index(item) > mylist.index('almond'):
            resultrecipe.append(item)
    resultrecipe.append('almond')
    resultrecipe.sort()
    #print(len(resultrecipe))
    x=len(resultrecipe)
    '''
    for i in range(437):
        if data['Ingredients'].str.contains(resultrecipe[i]) is False:
            data.drop(indexNames , inplace=True)
    print(data)'''
    recipe = data["Recipe_Name"].tolist()
    #photo = data["Recipe_Photo"].tolist()
    '''
    #print (data[['Recipe_Name', 'Ingredients']])
    #print(data.loc[data['Ingredients'].str.contains("egg") , 'Recipe_Name'])

    d = {}
    array=list
    for i in range(943):
        array = data.loc[data['Ingredients'].str.contains(mylist[i]) , 'Recipe_Name']
        d.setdefault(mylist[i], []).append(array)
    #print(d) 
    drecipe = {}
    array1=list
    for i in range(12351):
        array1 = data.loc[data['Recipe_Name'] == recipe[i], 'Ingredients']
        ingr= array1[i].split(',')
        drecipe.setdefault(recipe[i], []).append(ingr)
    #print(drecipe)

    review = data["Review_Count"].tolist()
    author = data["Author"].tolist()
    prepare = data["Prepare_Time"].tolist()
    cook = data["Cook_Time"].tolist()
    total = data["Total_Time"].tolist()
    directions = data["Directions"].tolist()

    #print(tuple(d.items())[14])
    '''
    drecipe = {}
    array1=list
    for i in range(1000):
        array1 = data.loc[data['Recipe_Name'] == recipe[i], 'Ingredients']
        ingr= array1[i].split(',')
        drecipe.setdefault(recipe[i], []).append(ingr)
    #print(drecipe)
    from sklearn.preprocessing import MultiLabelBinarizer
    # Instantiate the binarizer
    mlb = MultiLabelBinarizer()

    df = data.apply(lambda x: x["Ingredients"].split(","), axis=1)
    # Transform to a binary array
    array_out = mlb.fit_transform(df)

    df_out = pd.DataFrame(data=array_out, columns=mlb.classes_)

    #print(df_out)
    #merged = pd.concat([data['Recipe_Name'] , df_out], axis='columns')
    #print(mylist)

    out = ['xyz']
    for item in mylist:
        if mylist.index(item) < mylist.index('almond'):
            str = item.split(',')
            out.extend(str)
    out.remove('xyz')
    #print(len(out))
    y=len(out)
    #print(len(mylist))

    ml=df_out.iloc[:, 0:y]
    final = df_out.drop(ml, axis='columns' , inplace=True)

    merged = pd.concat([data['Recipe_Name'] , df_out], axis='columns') # recipe and all ingredients dataframe

    #finalmerged = merged.drop('Ingredients', axis='columns' )
    #print(merged.tail())

    #new_ing = [ 'potato','onion','tomato','broccoli']
    length = len(new_ing)
    naya = x - length

    list1 = [None] * naya        #populate list, length n with n entries "None"

    for i in range(length):
        list1.insert(i,new_ing[i])         #redefine list as the last n elements of list
    #print(len(list1))

    array = [None] * x 
    for i in range(len(resultrecipe)):
        #print(i)
        if list1[i] != None:
            stri = list1[i]
            location = resultrecipe.index(stri)
        if stri in resultrecipe:
            array[location] = 1
        else:
            array[location] = 0  #if you want to add dialog box for ingredient not

    for i in range(x):
        if array[i] == None:
            array[i] = 0
    #print(array)   #final vector form of given ingredients

    #x1 = merged.iloc[:, merged.columns != 'Recipe_Name'].as_matrix()

    #print(array)
    #print(x1[11])

    def cos_sim(a , b):
        """Takes 2 vectors a, b and returns the cosine similarity according 
        to the definition of the dot product
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    #print(cos_sim(array, x1[11]))
    forcalc = []
    resultmaybe = []
    z=[]
    for i in range(1000):
        forcalc = merged.iloc[i, ~merged.columns.isin(['Recipe_Name'])]
        z1 = cos_sim(array, forcalc)
        z.append(z1)
    
    cloud = dict( zip( recipe, z))
    #print( recipe)
    from collections import OrderedDict
    d_sorted_by_value = OrderedDict(sorted(cloud.items(), key=lambda x: x[1], reverse=True))

    for key, value in dict(d_sorted_by_value).items():
            if value == 0.0:
                del d_sorted_by_value[key]

    #print( d_sorted_by_value)  #dictionary with similar recipe and similarity
    #print( len(d_sorted_by_value))
    review = []
    author = []
    lastrecipe = []
    directions = []
    preptime = []
    cooktime = []
    totaltime = []
    ingred = []
    pic = []
    for key, value in dict(d_sorted_by_value).items():
        lastrecipe.append(key)
        review.append(data.loc[data['Recipe_Name'] == key, 'Review_Count'])
        author.append(data.loc[data['Recipe_Name'] == key, 'Author'])
        preptime.append(data.loc[data['Recipe_Name'] == key, 'Prepare_Time'])
        cooktime.append(data.loc[data['Recipe_Name'] == key, 'Cook_Time'])
        totaltime.append(data.loc[data['Recipe_Name'] == key, 'Total_Time'])
        ingred.append(data.loc[data['Recipe_Name'] == key, 'Ingredients'])
        directions.append(data.loc[data['Recipe_Name'] == key, 'Directions'])
        pic.append(data.loc[data['Recipe_Name'] == key, 'Recipe_Photo'])
        
    '''for i in range(len(d_sorted_by_value)):
    print("Recipe name: " + lastrecipe[i])
    print("Review count: " + review[i].to_string())
    print("Author name: " + author[i].to_string())'''

    for i in range(len(d_sorted_by_value)):
        review[i] = review[i].to_string(index=False)
        author[i] = author[i].to_string(index=False)
        preptime[i] = preptime[i].to_string(index=False)
        cooktime[i] = cooktime[i].to_string(index=False)
        totaltime[i] = totaltime[i].to_string(index=False)
        ingred[i] = ingred[i].to_string(index=False)
        directions[i] = directions[i].to_string(index=False)
        pic[i] = pic[i].to_string(index=False)

    

    #table_data = [[lastrecipe[i], author[i], review[i], ingred[i]]  for i in range(0, len(lastrecipe))]
    lst = zip(lastrecipe, author, review,preptime, cooktime, totaltime, ingred, directions, pic)
    return render_template("recipemy.html", lst=lst)
         

@app.route("/home/restaurant")
def restaurant():

         return render_template("restaurant.html")


@app.route("/home/restaurant/map3")
def map3():

         return render_template("map3.html")


@app.route("/home/restaurant/jimbu")
def jimbu():

         return render_template("jimbu.html")

@app.route("/home/restaurant/tukche")
def tukche():

         return render_template("tukche.html")

@app.route("/home/restaurant/lete")
def lete():

         return render_template("lete.html")

@app.route("/home/restaurant/hangkok")
def hangkok():

         return render_template("hangkok.html")

@app.route("/home/restaurant/picnic")
def picnic():

         return render_template("picnic.html")

@app.route("/home/restaurant/kimchee")
def kimchee():

         return render_template("kimchee.html")
         
@app.route("/home/restaurant/baanthai")
def baanthai():

         return render_template("baanthai.html")

@app.route("/home/restaurant/yingyang")
def yingyang():

         return render_template("yingyang.html")

@app.route("/home/restaurant/mangochilli")
def mangochilli():

         return render_template("mangochilli.html")

@app.route("/home/restaurant/ladolcevita")
def ladolcevita():

         return render_template("ladolcevita.html")

@app.route("/home/restaurant/littleitaly")
def littleitaly():

         return render_template("littleitaly.html")

@app.route("/home/restaurant/blackwater")
def blackwater():

         return render_template("blackwater.html")

@app.route("/home/restaurant/hyderabad")
def hyderabad():

         return render_template("hyderabad.html")

@app.route("/home/restaurant/aangan")
def aangan():

         return render_template("aangan.html")

@app.route("/home/restaurant/kolkata")
def kolkata():

         return render_template("kolkata.html")




@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))


def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/profile_pics', picture_fn)

    output_size = (125, 125)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn


@app.route("/account", methods=['GET', 'POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Your account has been updated!', 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static', filename='profile_pics/' + current_user.image_file)
    return render_template('account.html', title='Account',
                           image_file=image_file, form=form)


@app.route("/post/new", methods=['GET', 'POST'])
@login_required
def new_post():
    form = PostForm()
    if form.validate_on_submit():
        post = Post(title=form.title.data, content=form.content.data, author=current_user)
        db.session.add(post)
        db.session.commit()
        flash('Your post has been created!', 'success')
        return redirect(url_for('dashboard'))
    return render_template('create_post.html', title='New Post',
                           form=form, legend='New Post')


@app.route("/post/<int:post_id>")
def post(post_id):
    post = Post.query.get_or_404(post_id)
    return render_template('post.html', title=post.title, post=post)


@app.route("/post/<int:post_id>/update", methods=['GET', 'POST'])
@login_required
def update_post(post_id):
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        abort(403)
    form = PostForm()
    if form.validate_on_submit():
        post.title = form.title.data
        post.content = form.content.data
        db.session.commit()
        flash('Your post has been updated!', 'success')
        return redirect(url_for('post', post_id=post.id))
    elif request.method == 'GET':
        form.title.data = post.title
        form.content.data = post.content
    return render_template('create_post.html', title='Update Post',
                           form=form, legend='Update Post')


@app.route("/post/<int:post_id>/delete", methods=['POST'])
@login_required
def delete_post(post_id):
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        abort(403)
    db.session.delete(post)
    db.session.commit()
    flash('Your post has been deleted!', 'success')
    return redirect(url_for('dashboard'))


@app.route("/user/<string:username>")
def user_posts(username):
    page = request.args.get('page', 1, type=int)
    user = User.query.filter_by(username=username).first_or_404()
    posts = Post.query.filter_by(author=user)\
        .order_by(Post.date_posted.desc())\
        .paginate(page=page, per_page=5)
    return render_template('user_posts.html', posts=posts, user=user)


def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message('Password Reset Request',
                  sender='noreply@demo.com',
                  recipients=[user.email])
    msg.body = f'''To reset your password, visit the following link:
{url_for('reset_token', token=token, _external=True)}

If you did not make this request then simply ignore this email and no changes will be made.
'''
    mail.send(msg)


@app.route("/reset_password", methods=['GET', 'POST'])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RequestResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        send_reset_email(user)
        flash('An email has been sent with instructions to reset your password.', 'info')
        return redirect(url_for('login'))
    return render_template('reset_request.html', title='Reset Password', form=form)


@app.route("/reset_password/<token>", methods=['GET', 'POST'])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    user = User.verify_reset_token(token)
    if user is None:
        flash('That is an invalid or expired token', 'warning')
        return redirect(url_for('reset_request'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user.password = hashed_password
        db.session.commit()
        flash('Your password has been updated! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('reset_token.html', title='Reset Password', form=form)
