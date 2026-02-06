# n8n Setup Guide (Linux/Docker)

Since you plan to deploy this on your Linux machine, the cleanest way to run n8n is via Docker. This keeps it isolated and easy to manage.

## 1. Install Docker & Docker Compose (If not installed)
```bash
sudo apt update
sudo apt install docker.io docker-compose -y
sudo usermod -aG docker $USER
# Log out and back in for permissions to take effect
```

## 2. Create an n8n Docker Compose File
Create a folder for n8n (e.g., `~/n8n`) and a `docker-compose.yml` file inside it:

```yaml
version: "3"
services:
  n8n:
    image: docker.n8n.io/n8nio/n8n
    ports:
      - "5678:5678"
    environment:
      - N8N_HOST=n8n.local
    volumes:
      - ~/.n8n:/home/node/.n8n
      # IMPORTANT: Map the folder where your Python scripts live
      - /path/to/your/PaddleOCR:/opt/PaddleOCR
      # IMPORTANT: Map the data folder where PDFs arrive
      - /path/to/your/data:/opt/data
    restart: always
```
*Replace `/path/to/your/...` with the actual paths on your Linux machine.*

## 3. Start n8n
```bash
docker-compose up -d
```
Visit `http://localhost:5678` in your browser.

## 4. Building the Workflow
1.  **Trigger**: Use a **Local File Trigger** node.
    -   Watch folder: `/opt/data` (mapped volume).
    -   Event: "On File Created".
2.  **Execute Python**: Use an **Execute Command** node.
    -   Command: `python3 /opt/PaddleOCR/execution/workflow_runner.py --file "{{ $json.path }}"`
    -   *Note: You might need to install python dependencies inside the n8n container or use a custom image if you want n8n to run the script directly. Alternatively, n8n can call a webhook to a separate service.*

### Easy Mode (SSH/Execute remotely)
If you don't want to mess with Docker containers running Python, n8n can use an **SSH** node to connect to your *host machine* and run the command there:
1.  **Trigger**: Watch folder.
2.  **SSH**: Connect to `localhost` (or host IP).
3.  **Command**: `cd /home/user/PaddleOCR && python3 execution/workflow_runner.py --file "/path/to/new/file.pdf"`
