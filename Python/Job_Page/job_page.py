from flask import Flask, render_template, request, redirect, url_for, flash
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SelectField, SubmitField
from wtforms.validators import DataRequired, Length

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

# Define job categories
JOB_CATEGORIES = [
    ('web-development', 'Web Development'),
    ('mobile-development', 'Mobile Development'),
    ('design-creative', 'Design & Creative'),
    ('writing-translation', 'Writing & Translation'),
    ('data-science-analytics', 'Data Science & Analytics'),
    ('customer-service', 'Customer Service'),
    ('sales-marketing', 'Sales & Marketing'),
    ('admin-support', 'Admin Support')
]

# Define job posting form
class JobPostingForm(FlaskForm):
    title = StringField('Job Title', validators=[DataRequired(), Length(min=5, max=100)])
    category = SelectField('Job Category', choices=JOB_CATEGORIES, validators=[DataRequired()])
    description = TextAreaField('Job Description', validators=[DataRequired(), Length(min=100)])
    skills = StringField('Required Skills (comma separated)', validators=[DataRequired()])
    budget = StringField('Budget', validators=[DataRequired()])
    submit = SubmitField('Post Job')

# Sample job data (replace with database/persistent storage in a real application)
jobs = []

@app.route('/', methods=['GET', 'POST'])
def index():
    form = JobPostingForm()
    if form.validate_on_submit():
        job = {
            'title': form.title.data,
            'category': form.category.data,
            'description': form.description.data,
            'skills': form.skills.data.split(','),
            'budget': form.budget.data
        }
        jobs.append(job)
        flash('Job posted successfully!', 'success')
        return redirect(url_for('index'))
    return render_template('index.html', form=form, jobs=jobs)

@app.route('/job/<int:job_id>')
def job_details(job_id):
    if 0 <= job_id < len(jobs):
        job = jobs[job_id]
        return render_template('job_details.html', job=job)
    else:
        flash('Invalid job ID.', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
