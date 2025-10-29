# Google Docs API Setup Instructions

Follow these steps to set up Google Docs API authentication:

## Step 1: Create a Google Cloud Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" at the top, then click "New Project"
3. Name your project (e.g., "Resume Agent") and click "Create"

## Step 2: Enable Google Docs API

1. In your project, go to "APIs & Services" > "Library"
2. Search for "Google Docs API"
3. Click on it and press "Enable"

## Step 3: Configure OAuth Consent Screen

1. Go to "APIs & Services" > "OAuth consent screen"
2. Choose "External" user type and click "Create"
3. Fill in the required fields:
    - App name: "Resume Agent" (or any name you prefer)
    - User support email: Select your email
    - Developer contact email: Enter your email (yang.lu.queens@gmail.com)
4. Click "Save and Continue"
5. On the Scopes page, click "Save and Continue" (no need to add scopes)
6. **IMPORTANT**: On the "Test users" page:
    - Click "+ ADD USERS"
    - Enter your email: **yang.lu.queens@gmail.com**
    - Click "Add"
    - Click "Save and Continue"
7. Review the summary and click "Back to Dashboard"

## Step 4: Create OAuth 2.0 Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth client ID"
3. Choose "Desktop app" as the application type
4. Name it (e.g., "Resume Agent Desktop")
5. Click "Create"

## Step 5: Download Credentials

1. After creating the OAuth client ID, click the download button (⬇) next to your credential
2. Save the downloaded file as `credentials.json` in this project directory (`c:\Users\yang\Documents\GitHub\resume-agent`)

## Step 6: Run the Application

1. Run the application:

    ```
    python app.py
    ```

2. The first time you run it:

    - A browser window will open
    - Sign in with your Google account
    - Grant the requested permissions (read-only access to Google Docs)
    - The browser will show "The authentication flow has completed"
    - A `token.json` file will be created automatically

3. For subsequent runs, authentication will be automatic using the saved `token.json`

## Troubleshooting

### "Access blocked: Resume agent has not completed the Google verification process"

This error means your email is not added as a test user. To fix:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select your project
3. Go to "APIs & Services" > "OAuth consent screen"
4. Scroll down to "Test users" section
5. Click "ADD USERS"
6. Enter your email: **yang.lu.queens@gmail.com**
7. Click "Add" and "Save"
8. Wait a few minutes for the changes to propagate
9. Try running `python app.py` again

### Other Common Issues

-   **"credentials.json not found"**: Make sure you downloaded the credentials file and placed it in the correct directory
-   **Permission errors**: Delete `token.json` and run again to re-authenticate
-   **Browser doesn't open**: The authentication URL will be printed in the terminal - copy and paste it into your browser

## Step 7: Install Python Dependencies

1. Install the required Python packages:

    ```
    pip install -r requirements.txt
    ```

2. (Optional) Set up environment variables for local development:

    ```
    cp .env.example .env
    # Edit .env with your actual API key
    ```

## Step 8: Make Google Doc Public (Required for Dynamic Updates)

**Important:** For dynamic resume updates, you must share your Google Doc publicly:

1. Open your resume Google Doc in the browser
2. Click **Share** (top right)
3. Set permissions to **"Anyone with the link"** > **"Viewer"**
4. Click **"Done"**
5. Your doc will now accept export requests without authentication

## Step 9: Run the Chatbot Application

1. Run the application:

    ```
    python app.py
    ```

2. The application will fetch the resume content from your public Google Doc.

3. After loading the resume, a Gradio interface will launch in your browser at `http://127.0.0.1:7860/` (or similar).

4. Use the chat interface to ask questions about the candidate. The chatbot will only provide answers based on the resume content.

## Step 10: Deploy to Hugging Face Spaces (Optional Free Hosting)

Hugging Face Spaces provides free hosting for your chatbot with dynamic resume updates. Great resume showcase!

### Prerequisites

-   Completed Steps 7-8 (dependencies installed, doc made public)
-   Hugging Face account ([huggingface.co](https://huggingface.co))

### Deployment Steps

1. **Create a New Space:**

    - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
    - Click **"Create new Space"**
    - Name: `resume-agent-chatbot` (or your choice)
    - SDK: **Gradio**
    - Visibility: Public
    - Click **"Create Space"**

2. **Upload Your Files:**

    - In your new Space, clone the repository locally or use the online editor
    - Add these files:
        - `app.py`
        - `requirements.txt`
    - Commit the changes

3. **Set Repository Secrets (if needed):**

    - If you want to hide the Z.AI API key, add it as a repository secret:
        - Go to Space Settings > Repository Secrets
        - Add `ZAI_API_KEY` = your API key
    - Optionally hide the Document ID:
        - Add `DOCUMENT_ID` = your Google Doc ID (e.g., `1y8qge7gB9otoVnTxj88ZPyYRaVfxsyFNBWF3tITcaPk`)
    - Customize AI behavior (optional):
        - Add `PROMPT_TEMPLATE` = your custom prompt template (see .env.example for format)
    - The code will use environment variables (`os.getenv()`) if available, falling back to hardcoded defaults

4. **Wait for Build & Enjoy:**
    - HF will install dependencies and launch your app
    - Your Space URL: `https://[username].huggingface.co/spaces/resume-agent-chatbot`
    - Click the URL to see your live chatbot!

### Resume Bullet Point

Add this to your resume:

```
Deployed AI-powered resume chatbot on Hugging Face Spaces with dynamic Google Docs integration
```

### Troubleshooting HF Spaces

-   **Build failures:** Check app.py syntax, ensure all imports work
-   **Resume not loading:** Verify your Google Doc is set to "Anyone with the link" > "Viewer"
-   **API rate limits:** If Z.AI API limits occur, the free tier should suffice for demo usage

## AWS Deployment Notes

For hosting on AWS (or any server environment), you'll need to use Google Service Account authentication instead of the user OAuth flow. The `token.json` file cannot be reused because it's tied to individual user sessions and requires interactive login.

### Step 9: Set Up Google Service Account (For AWS Deployment)

1. **Create a Service Account:**

    - Go to [Google Cloud Console](https://console.cloud.google.com/) > IAM & Admin > Service Accounts
    - Click "+ Create Service Account"
    - Name: "resume-agent" (or any name)
    - Description: "Service account for resume agent chatbot"
    - Click "Create and Continue"
    - Skip roles (optional) and click "Done"

2. **Create Key for Service Account:**

    - In the Service Accounts list, click on your new service account
    - Go to the "Keys" tab
    - Click "Add Key" > "Create new key" > JSON format
    - Download the JSON key file (this is your `service-account.json`)

3. **Share the Google Doc:**

    - Open your resume Google Doc
    - Click "Share" (top right)
    - Add the service account email (ends with @your-project.iam.gserviceaccount.com) as an "Editor"
    - Click "Send"

4. **Configure Environment Variables on AWS:**
    - Set `ZAI_API_KEY` to your Z.AI API key
    - Set `USE_SERVICE_ACCOUNT=true` (optional, as the code auto-detects the file)
    - Upload `service-account.json` to your AWS environment securely

### AWS Deployment Steps

1. **Deploy to AWS EC2/ECS/Lambda/etc:**

    - Upload your code to AWS (avoid committing `credentials.json`, `token.json`, or `service-account.json`)
    - Install dependencies with `pip install -r requirements.txt`
    - Set environment variables:
        ```
        ZAI_API_KEY=your_zai_api_key_here
        USE_SERVICE_ACCOUNT=true  # optional
        ```

2. **For Production Web Hosting:**
    - Consider using Gradio's share=True or host it behind a reverse proxy
    - For persistent hosting, use AWS EC2 or ECS
    - For scaling, consider Lambda with API Gateway

### Local vs Server Differences

-   **Local Development:** Uses OAuth (interactive login, `credentials.json` + `token.json`)
-   **Server/Production:** Uses Service Account (automated, `service-account.json`)

## Security Notes

-   `credentials.json` contains your OAuth client ID and secret - keep it private
-   `token.json` contains your access token - keep it private
-   Z.AI API key should be kept private; use environment variables in production
-   Add sensitive files to `.gitignore` to avoid committing them to version control
