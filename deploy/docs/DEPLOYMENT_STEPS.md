# Deployment Steps for PD Automations

This document is for DevOps/Infra teams to deploy the PD Automations Flask application using Docker and Kubernetes.

---

## 1. Prerequisites
- Access to the code repository (GitHub, GitLab, etc.)
- Docker installed
- Kubernetes cluster access (kubectl configured)
- Access to organization’s Docker registry (Docker Hub, AWS ECR, etc.)

---

## 2. Build and Push Docker Image
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd pd_automations
   ```
2. **Build the Docker image:**
   ```bash
   docker build -f deploy/Dockerfile -t <registry>/<project>:<tag> .
   ```
   - Note: Keep the build context as repo root (the final dot ".") so requirements and app files are available to COPY commands.
3. **Push the Docker image:**
   ```bash
   docker push <registry>/<project>:<tag>
   ```
4. **Update the image name** in `deploy/k8s/k8s-deployment.yaml`:
   ```yaml
   image: <registry>/<project>:<tag>
   ```

### Optional: Run Locally with Docker
```bash
docker run --rm \
  -e SESSION_SECRET=your_session_secret \
  -e OPENAI_API_KEY=your_openai_key \
  -p 5050:5050 \
  <registry>/<project>:<tag>
```
Then open http://localhost:5050

---

## 3. Prepare Kubernetes Secrets
Create secrets for required environment variables (do NOT commit secrets to Git):

```bash
kubectl create secret generic session-secret --from-literal=secret=<SESSION_SECRET_VALUE>
kubectl create secret generic openai-secret --from-literal=api-key=<OPENAI_API_KEY>
```

---

## 4. Deploy to Kubernetes
1. **Apply the deployment and service manifests:**
   ```bash
   kubectl apply -f deploy/k8s/
   ```
2. **Verify the deployment:**
   ```bash
   kubectl get pods
   kubectl get service pd-automations-service
   ```
   - Wait for the EXTERNAL-IP to be assigned, then access the app in your browser.

---

## 5. Persistent Storage (Optional)
If persistent storage is required for uploads/output, modify the deployment to use PersistentVolumeClaims instead of `emptyDir`.

---

## 6. Environment Variables Required
| Variable Name     | Description        | How to Provide         |
|-------------------|-------------------|------------------------|
| SESSION_SECRET    | Flask session key | Kubernetes secret      |
| OPENAI_API_KEY    | OpenAI API Key    | Kubernetes secret      |

Add any additional secrets or configuration as needed.

---

## 7. Troubleshooting & Logs
- Check pod logs:
  ```bash
  kubectl logs deployment/pd-automations
  ```
- Check service and pod status:
  ```bash
  kubectl get pods
  kubectl get services
  ```

---

## 8. Contact
For questions about the application or deployment, contact the development team.

---

*This document is for deployment purposes only. For application usage, see the main README.md.*
