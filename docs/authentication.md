# Authentication Module

## Overview
This module handles user registration, login, and authentication tokens.  
It ensures that only verified users can access the system.

## Features
- User registration with email and password
- Secure password hashing
- JWT-based authentication
- Role-based access (admin, standard user)

## Endpoints
- `POST /auth/register` → Create new user
- `POST /auth/login` → Authenticate user
- `GET /auth/me` → Get current user profile