# Meeting Minutes

## Summary

The weekly engineering sync focused on the backend migration to Kubernetes, with a cutover scheduled for March 15. The team confirmed that the deployment will not proceed until the staging cluster is ready. Key technical details, including migration time and rate limits, were reviewed, and a rollback runbook was assigned to Marcus.

## Minutes

### Deployment and Kubernetes Migration

- The backend is moving to Kubernetes with a cutover date of March 15.
- The staging cluster is not ready, so the team will not ship before it is.
- The Postgres migration takes about 40 minutes.
- The API rate limit remains at 1200 requests per minute.
- Local deployment uses Docker Compose and CI uses GitHub actions.

### Monitoring and Documentation

- Switching monitoring to Grafana was suggested but not agreed upon.
- The OAuth documentation needs to be updated, but no owner has been assigned.

