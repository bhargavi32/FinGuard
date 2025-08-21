# FinGuard Deployment Guide 🚀

## Quick Deployment Options

### Option 1: Simple Local Development (Fastest)

```bash
# 1. Create sample data
python scripts/setup_sample_data.py

# 2. Start API server
python simple_api_server.py

# 3. Access API documentation
# Visit: http://localhost:8000/docs
```

**Perfect for**: Development, testing, demos

### Option 2: Docker Compose (Recommended for Local)

```bash
# Simple version
docker-compose -f docker-compose.simple.yml up -d

# Full stack with database
docker-compose up -d
```

**Perfect for**: Local testing with full stack

### Option 3: Kubernetes (Production)

```bash
# Deploy to existing cluster
kubectl apply -f deployment/kubernetes/

# Or use Helm (if available)
helm install finguard ./charts/finguard
```

**Perfect for**: Production deployment

### Option 4: AWS Cloud (Enterprise)

```bash
# Deploy infrastructure
cd deployment/aws/terraform
terraform init
terraform apply

# Deploy application
# Use CI/CD pipeline or manual deployment
```

**Perfect for**: Cloud production environment

## Testing Your Deployment

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Fraud Prediction Test
```bash
curl -X POST "http://localhost:8000/fraud/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 1000.5,
    "Amount": 150.75,
    "V1": 0.144,
    "V2": -0.359,
    "V3": 1.123,
    "V4": -0.267,
    "V5": 0.567,
    "V6": -1.234,
    "V7": 0.890,
    "V8": -0.456,
    "V9": 1.789,
    "V10": -0.123
  }'
```

### 3. Recommendations Test
```bash
curl -X POST "http://localhost:8000/recommendations/" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "transaction_history": [
      {
        "Time": 1000,
        "Amount": 100,
        "V1": 0.1, "V2": -0.1, "V3": 0.2, "V4": -0.2,
        "V5": 0.3, "V6": -0.3, "V7": 0.4, "V8": -0.4,
        "V9": 0.5, "V10": -0.5
      }
    ],
    "num_recommendations": 3
  }'
```

## Environment Variables

### Required Variables
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Database (if using full stack)
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=finguard
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=finguard
```

### Optional Variables
```bash
# Security
SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30

# AWS (for cloud deployment)
AWS_REGION=us-east-1
AWS_S3_BUCKET=finguard-models

# Monitoring
MLFLOW_TRACKING_URI=http://localhost:5000
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Change port in environment or kill existing process
   lsof -ti:8000 | xargs kill -9
   ```

2. **Missing Dependencies**
   ```bash
   # Install basic requirements
   pip install fastapi uvicorn pydantic pandas scikit-learn
   ```

3. **Docker Issues**
   ```bash
   # Rebuild containers
   docker-compose down
   docker-compose build --no-cache
   docker-compose up -d
   ```

4. **Database Connection**
   ```bash
   # Check MySQL service
   docker-compose logs mysql
   ```

### Logs and Monitoring

```bash
# View application logs
docker-compose logs finguard-api

# Check all services
docker-compose logs

# Monitor resource usage
docker stats
```

## Production Checklist

### Security
- [ ] Change default passwords
- [ ] Configure TLS/SSL certificates
- [ ] Set up firewall rules
- [ ] Enable authentication
- [ ] Configure secrets management

### Performance
- [ ] Set appropriate resource limits
- [ ] Configure auto-scaling
- [ ] Set up load balancing
- [ ] Enable caching
- [ ] Configure database optimization

### Monitoring
- [ ] Set up health checks
- [ ] Configure alerting
- [ ] Enable log aggregation
- [ ] Set up metrics collection
- [ ] Configure backup procedures

### Compliance
- [ ] Enable audit logging
- [ ] Configure data retention
- [ ] Set up access controls
- [ ] Document procedures
- [ ] Test disaster recovery

## Support

For issues or questions:
1. Check logs first: `docker-compose logs`
2. Review health endpoints: `/health`
3. Check API documentation: `/docs`
4. Review configuration files
5. Contact: team@finguard.ai

---

**🎉 You're now ready to deploy FinGuard! Choose the deployment option that best fits your needs and follow the steps above.**
