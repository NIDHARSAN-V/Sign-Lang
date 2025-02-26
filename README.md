# Text to Sign Language Project

## Overview

The Text to Sign Language project aims to convert text input into sign language animations. This project utilizes HTML for the front end interface, Django for the back end server, and Blender for generating sign language animations.

## Features

- **Text Input:** Users can input text phrases or sentences.
- **Sign Language Conversion:** The input text is converted into sign language animations.
- **Playback Control:** Users can play, pause, rewind, and adjust the speed of the sign language animations.
- **Customization:** Users can choose different sign language styles and animation preferences.
- **Export:** Option to export sign language animations in various formats.

## Technologies Used

- **Front End:** HTML, CSS, JavaScript
- **Back End:** Django, Python
- **Animation:** Blender, Python scripting

## Getting Started

To run the project locally, follow these steps:

1. Clone this repository to your local machine.
2. Install Python and Django if not already installed.
3. Set up a virtual environment and activate it.
4. Install the required Python dependencies using `pip install -r requirements.txt`.
5. Install Blender and ensure it is added to the system PATH.
6. Start the Django development server using `python manage.py runserver`.
7. Access the application in your web browser at `http://localhost:8000`.

## Project Structure

text-to-sign-language/
│
├── backend/
│ ├── manage.py
│ └── text_to_sign_language/
│ ├── settings.py
│ ├── urls.py
│ └── ...
│
├── frontend/
│ ├── index.html
│ ├── style.css
│ └── script.js
│
├── animations/
│ ├── sign_language.blend
│ └── generate_animation.py
│
└── README.md


- **backend:** Contains Django backend code.
- **frontend:** Contains HTML, CSS, and JavaScript files for the front end.
- **animations:** Contains Blender project files for sign language animations and Python script for animation generation.

## Contributing

Contributions to this project are welcome! Feel free to open issues or pull requests to suggest improvements or report bugs.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
