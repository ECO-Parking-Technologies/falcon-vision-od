# ECO Parking API (GraphQL)

> Extracted from https://app.ecoparkingtechnologies.com/api/documentation/index.html (SpectaQL dump). Machine-readable version: `ECO-Parking-API.json`.

## What is GraphQL?

The ECO Parking API is a GraphQL API. It allows for more efficient retrieval of data than REST by enabling you to fetch multiple, nested resources in a single request.

GraphQL is a query language for APIs that allows clients to request exactly the data they need, making it possible to get all required data in a limited number of requests. The GraphQL data (fields) can be described in the form of types, allowing clients to use client-side GraphQL libraries to consume the API and avoid manual parsing.

As mentioned above, there are client-side libraries for virtually all modern programming languages. However, we also provide an API Explorer that will help you build queries and run them on your live data, completely in your browser and without the need to install anything. Once registered as a user in the ECO Parking Portal, go to API Explorer

API Endpoints

```
https://api.ecoparkingtechnologies.com/graphql
```

Create a token by going to Personal API Tokens . You will use this OAuth2 refresh token to create OAuth2 access tokens.

The OAuth2 server is located at: https://identity.ecoparkingtechnologies.com/token

You should only ask for a new token if the access token has expired or you want to refresh the claims contained in the ID token. For example, it's bad practice to call the endpoint to get a new access token every time you call an API. There are rate limits that will throttle the number of requests to this endpoint that can be executed using the same token from the same IP.

To exchange the refresh token you received during API token creation for a new access token, make a POST request to the /token endpoint in the Authentication API, using grant_type=refresh_token .

```
curl --request POST \
--url 'https://identity.ecoparkingtechnologies.com/token' \
--header 'content-type: application/x-www-form-urlencoded' \
--data grant_type=refresh_token \
--data 'client_id=EcoParking.ClientApi' \
--data refresh_token=YOUR_API_TOKEN
```

The response will include a new access token, its type, its lifetime (in seconds), and the granted scopes.

```
{
"access_token": "eyJ...MoQ",
"expires_in": 1200,
"scope": "offline_access",
"id_token": "eyJ...0NE",
"token_type": "Bearer"
}
```

When using this token, you will be accessing the API as the user the token was created by. It is important to protect this API token and not to expose it publicly. If it is exposed, delete it immediately and create a new one.

Let's look at some simple example queries to get a feel for how to interact with the API to get some commonly requested data.

```
query GetSites {
  sites {
    nodes {
      displayName
      id
    }
  }
}
```

will return a response similar similar to:

```
{
  "data": {
    "sites": {
      "nodes": [
        {
          "displayName": "9th Avenue West Garage",
          "id": "3"
        },
        {
          "displayName": "10th Avenue West Garage",
          "id": "5"
        }
      ]
    }
  }
}
```

```
query GetSiteLevelAvailability {
  latestSiteLevelParkingUsages(condition: {siteId: 15}) {
    nodes {
      availableCount
      occupiedCount
      timestamp
      siteLevel {
        displayName
      }
    }
  }
}
```

will return a response similar similar to:

```
{
  "data": {
    "latestSiteLevelParkingUsages": {
      "nodes": [
        {
          "availableCount": 50,
          "occupiedCount": 58,
          "timestamp": "2023-05-22T19:13:00.873509+00:00",
          "siteLevel": {
            "displayName": "Level 1"
          }
        },
        {
          "availableCount": 50,
          "occupiedCount": 25,
          "timestamp": "2023-05-22T19:13:00.873509+00:00",
          "siteLevel": {
            "displayName": "Level 2"
          }
        },
        {
          "availableCount": 26,
          "occupiedCount": 8,
          "timestamp": "2023-05-22T19:13:00.873509+00:00",
          "siteLevel": {
            "displayName": "Level 3"
          }
        },
        {
          "availableCount": 0,
          "occupiedCount": 94,
          "timestamp": "2023-05-22T19:13:00.873509+00:00",
          "siteLevel": {
            "displayName": "Top Deck"
          }
        }
      ]
    }
  }
}
```

```
query GetMonthlyAverages {
  site(id: 15) {
    displayName
    siteLevelParkingAttributeUsageByHours(
      filter: {hour: {greaterThanOrEqualTo: "2022-02-01", lessThanOrEqualTo: "2022-02-28"}}
    ) {
      groupedAggregates(groupBy: [HOUR]) {
        sum {
          occupiedCount
          availableCount
        }
        keys
      }
    }
  }
}
```

will return a response similar similar to:

```
{
  "data": {
    "site": {
      "siteLevelSnapshotHistories": {
        "groupedAggregates": [
          {
            "sum": {
              "occupiedCount": "173.97",
              "availableCount": "137.03"
            },
            "keys": [
              "2023-06-03 00:00:00+00"
            ]
          },
          {
            "sum": {
              "occupiedCount": "172.76",
              "availableCount": "138.24"
            },
            "keys": [
              "2023-06-03 01:00:00+00"
            ]
          },
          {
            "sum": {
              "occupiedCount": "176.22",
              "availableCount": "134.78"
            },
            "keys": [
              "2023-06-03 02:00:00+00"
            ]
          },
          ...
        ]
      },
      "displayName": "6th Street Garage"
    }
  }
}
```

## Authentication

Create a token by going to Personal API Tokens . You will use this OAuth2 refresh token to create OAuth2 access tokens.

The OAuth2 server is located at: https://identity.ecoparkingtechnologies.com/token

You should only ask for a new token if the access token has expired or you want to refresh the claims contained in the ID token. For example, it's bad practice to call the endpoint to get a new access token every time you call an API. There are rate limits that will throttle the number of requests to this endpoint that can be executed using the same token from the same IP.

To exchange the refresh token you received during API token creation for a new access token, make a POST request to the /token endpoint in the Authentication API, using grant_type=refresh_token .

```
curl --request POST \
--url 'https://identity.ecoparkingtechnologies.com/token' \
--header 'content-type: application/x-www-form-urlencoded' \
--data grant_type=refresh_token \
--data 'client_id=EcoParking.ClientApi' \
--data refresh_token=YOUR_API_TOKEN
```

The response will include a new access token, its type, its lifetime (in seconds), and the granted scopes.

```
{
"access_token": "eyJ...MoQ",
"expires_in": 1200,
"scope": "offline_access",
"id_token": "eyJ...0NE",
"token_type": "Bearer"
}
```

When using this token, you will be accessing the API as the user the token was created by. It is important to protect this API token and not to expose it publicly. If it is exposed, delete it immediately and create a new one.

## Example Queries

Let's look at some simple example queries to get a feel for how to interact with the API to get some commonly requested data.

```
query GetSites {
  sites {
    nodes {
      displayName
      id
    }
  }
}
```

will return a response similar similar to:

```
{
  "data": {
    "sites": {
      "nodes": [
        {
          "displayName": "9th Avenue West Garage",
          "id": "3"
        },
        {
          "displayName": "10th Avenue West Garage",
          "id": "5"
        }
      ]
    }
  }
}
```

```
query GetSiteLevelAvailability {
  latestSiteLevelParkingUsages(condition: {siteId: 15}) {
    nodes {
      availableCount
      occupiedCount
      timestamp
      siteLevel {
        displayName
      }
    }
  }
}
```

will return a response similar similar to:

```
{
  "data": {
    "latestSiteLevelParkingUsages": {
      "nodes": [
        {
          "availableCount": 50,
          "occupiedCount": 58,
          "timestamp": "2023-05-22T19:13:00.873509+00:00",
          "siteLevel": {
            "displayName": "Level 1"
          }
        },
        {
          "availableCount": 50,
          "occupiedCount": 25,
          "timestamp": "2023-05-22T19:13:00.873509+00:00",
          "siteLevel": {
            "displayName": "Level 2"
          }
        },
        {
          "availableCount": 26,
          "occupiedCount": 8,
          "timestamp": "2023-05-22T19:13:00.873509+00:00",
          "siteLevel": {
            "displayName": "Level 3"
          }
        },
        {
          "availableCount": 0,
          "occupiedCount": 94,
          "timestamp": "2023-05-22T19:13:00.873509+00:00",
          "siteLevel": {
            "displayName": "Top Deck"
          }
        }
      ]
    }
  }
}
```

```
query GetMonthlyAverages {
  site(id: 15) {
    displayName
    siteLevelParkingAttributeUsageByHours(
      filter: {hour: {greaterThanOrEqualTo: "2022-02-01", lessThanOrEqualTo: "2022-02-28"}}
    ) {
      groupedAggregates(groupBy: [HOUR]) {
        sum {
          occupiedCount
          availableCount
        }
        keys
      }
    }
  }
}
```

will return a response similar similar to:

```
{
  "data": {
    "site": {
      "siteLevelSnapshotHistories": {
        "groupedAggregates": [
          {
            "sum": {
              "occupiedCount": "173.97",
              "availableCount": "137.03"
            },
            "keys": [
              "2023-06-03 00:00:00+00"
            ]
          },
          {
            "sum": {
              "occupiedCount": "172.76",
              "availableCount": "138.24"
            },
            "keys": [
              "2023-06-03 01:00:00+00"
            ]
          },
          {
            "sum": {
              "occupiedCount": "176.22",
              "availableCount": "134.78"
            },
            "keys": [
              "2023-06-03 02:00:00+00"
            ]
          },
          ...
        ]
      },
      "displayName": "6th Street Garage"
    }
  }
}
```

# Queries

## currentClientId

**Response:** Returns an Int

```graphql
query currentClientId {
  currentClientId
}
```

## currentUser

**Response:** Returns a User

```graphql
query currentUser {
  currentUser {
    nodeId
    id
    email
    givenName
    familyName
    createdTimestamp
    createdUserId
    createdClientId
    lastModifiedTimestamp
    lastModifiedUserId
    lastModifiedClientId
    subjectId
    isDeleted
    createdUser {
      ...UserFragment
    }
    lastModifiedUser {
      ...UserFragment
    }
    usersByCreatedUserId {
      ...UsersConnectionFragment
    }
    usersByLastModifiedUserId {
      ...UsersConnectionFragment
    }
    organizationsByCreatedUserId {
      ...OrganizationsConnectionFragment
    }
    organizationsByLastModifiedUserId {
      ...OrganizationsConnectionFragment
    }
    organizationUsersByUserId {
      ...OrganizationUsersConnectionFragment
    }
    organizationUsersByCreatedUserId {
      ...OrganizationUsersConnectionFragment
    }
    organizationUsersByLastModifiedUserId {
      ...OrganizationUsersConnectionFragment
    }
    parkingAttributesByCreatedUserId {
      ...ParkingAttributesConnectionFragment
    }
    parkingAttributesByLastModifiedUserId {
      ...ParkingAttributesConnectionFragment
    }
    parkingSpacesByCreatedUserId {
      ...ParkingSpacesConnectionFragment
    }
    parkingSpacesByLastModifiedUserId {
      ...ParkingSpacesConnectionFragment
    }
    parkingZonesByCreatedUserId {
      ...ParkingZonesConnectionFragment
    }
    parkingZonesByLastModifiedUserId {
      ...ParkingZonesConnectionFragment
    }
    sitesByCreatedUserId {
      ...SitesConnectionFragment
    }
    sitesByLastModifiedUserId {
      ...SitesConnectionFragment
    }
    siteLevelsByCreatedUserId {
      ...SiteLevelsConnectionFragment
    }
    siteLevelsByLastModifiedUserId {
      ...SiteLevelsConnectionFragment
    }
    organizationRolesByCreatedUserId {
      ...OrganizationRolesConnectionFragment
    }
    organizationRolesByLastModifiedUserId {
      ...OrganizationRolesConnectionFragment
    }
  }
}
```

## currentUserId

**Response:** Returns an Int

```graphql
query currentUserId {
  currentUserId
}
```

## currentUserOrganizationRole

**Response:** Returns a String

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `organizationId` | `Int` |  |

```graphql
query currentUserOrganizationRole($organizationId: Int) {
  currentUserOrganizationRole(organizationId: $organizationId)
}
```

## latestSiteLevelParkingAttributeParkingUsages

Description Reads and enables pagination through a set of LatestSiteLevelParkingAttributeParkingUsage .

**Response:** Returns a LatestSiteLevelParkingAttributeParkingUsagesConnection

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `first` | `Int` |  |
| `last` | `Int` |  |
| `offset` | `Int` |  |
| `before` | `Cursor` |  |
| `after` | `Cursor` |  |
| `orderBy` | `[LatestSiteLevelParkingAttributeParkingUsagesOrderBy!]` |  |
| `condition` | `LatestSiteLevelParkingAttributeParkingUsageCondition` |  |
| `filter` | `LatestSiteLevelParkingAttributeParkingUsageFilter` |  |

```graphql
query latestSiteLevelParkingAttributeParkingUsages(
  $first: Int,
  $last: Int,
  $offset: Int,
  $before: Cursor,
  $after: Cursor,
  $orderBy: [LatestSiteLevelParkingAttributeParkingUsagesOrderBy!],
  $condition: LatestSiteLevelParkingAttributeParkingUsageCondition,
  $filter: LatestSiteLevelParkingAttributeParkingUsageFilter
) {
  latestSiteLevelParkingAttributeParkingUsages(
    first: $first,
    last: $last,
    offset: $offset,
    before: $before,
    after: $after,
    orderBy: $orderBy,
    condition: $condition,
    filter: $filter
  ) {
    nodes {
      ...LatestSiteLevelParkingAttributeParkingUsageFragment
    }
    totalCount
  }
}
```

## latestSiteParkingAttributeParkingUsages

Description Reads and enables pagination through a set of LatestSiteParkingAttributeParkingUsage .

**Response:** Returns a LatestSiteParkingAttributeParkingUsagesConnection

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `first` | `Int` |  |
| `last` | `Int` |  |
| `offset` | `Int` |  |
| `before` | `Cursor` |  |
| `after` | `Cursor` |  |
| `orderBy` | `[LatestSiteParkingAttributeParkingUsagesOrderBy!]` |  |
| `condition` | `LatestSiteParkingAttributeParkingUsageCondition` |  |
| `filter` | `LatestSiteParkingAttributeParkingUsageFilter` |  |

```graphql
query latestSiteParkingAttributeParkingUsages(
  $first: Int,
  $last: Int,
  $offset: Int,
  $before: Cursor,
  $after: Cursor,
  $orderBy: [LatestSiteParkingAttributeParkingUsagesOrderBy!],
  $condition: LatestSiteParkingAttributeParkingUsageCondition,
  $filter: LatestSiteParkingAttributeParkingUsageFilter
) {
  latestSiteParkingAttributeParkingUsages(
    first: $first,
    last: $last,
    offset: $offset,
    before: $before,
    after: $after,
    orderBy: $orderBy,
    condition: $condition,
    filter: $filter
  ) {
    nodes {
      ...LatestSiteParkingAttributeParkingUsageFragment
    }
    totalCount
  }
}
```

## latestSiteParkingUsages

Description Reads and enables pagination through a set of LatestSiteParkingUsage .

**Response:** Returns a LatestSiteParkingUsagesConnection

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `first` | `Int` |  |
| `last` | `Int` |  |
| `offset` | `Int` |  |
| `before` | `Cursor` |  |
| `after` | `Cursor` |  |
| `orderBy` | `[LatestSiteParkingUsagesOrderBy!]` |  |
| `condition` | `LatestSiteParkingUsageCondition` |  |
| `filter` | `LatestSiteParkingUsageFilter` |  |

```graphql
query latestSiteParkingUsages(
  $first: Int,
  $last: Int,
  $offset: Int,
  $before: Cursor,
  $after: Cursor,
  $orderBy: [LatestSiteParkingUsagesOrderBy!],
  $condition: LatestSiteParkingUsageCondition,
  $filter: LatestSiteParkingUsageFilter
) {
  latestSiteParkingUsages(
    first: $first,
    last: $last,
    offset: $offset,
    before: $before,
    after: $after,
    orderBy: $orderBy,
    condition: $condition,
    filter: $filter
  ) {
    nodes {
      ...LatestSiteParkingUsageFragment
    }
    totalCount
  }
}
```

## node

Description Fetches an object given its globally unique ID .

**Response:** Returns a Node

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |

```graphql
query node($nodeId: ID!) {
  node(nodeId: $nodeId) {
    nodeId
  }
}
```

## nodeId

Description The root query type must be a Node to work well with Relay 1 mutations. This just resolves to query .

**Response:** Returns an ID!

```graphql
query nodeId {
  nodeId
}
```

## organization

**Response:** Returns an Organization

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `Int!` |  |

```graphql
query organization($id: Int!) {
  organization(id: $id) {
    nodeId
    id
    name
    displayName
    description
    createdTimestamp
    createdUserId
    createdClientId
    lastModifiedTimestamp
    lastModifiedUserId
    lastModifiedClientId
    isDeleted
    createdUser {
      ...UserFragment
    }
    lastModifiedUser {
      ...UserFragment
    }
    organizationUsers {
      ...OrganizationUsersConnectionFragment
    }
    sites {
      ...SitesConnectionFragment
    }
  }
}
```

## organizationByNodeId

Description Reads a single Organization using its globally unique ID .

**Response:** Returns an Organization

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |

```graphql
query organizationByNodeId($nodeId: ID!) {
  organizationByNodeId(nodeId: $nodeId) {
    nodeId
    id
    name
    displayName
    description
    createdTimestamp
    createdUserId
    createdClientId
    lastModifiedTimestamp
    lastModifiedUserId
    lastModifiedClientId
    isDeleted
    createdUser {
      ...UserFragment
    }
    lastModifiedUser {
      ...UserFragment
    }
    organizationUsers {
      ...OrganizationUsersConnectionFragment
    }
    sites {
      ...SitesConnectionFragment
    }
  }
}
```

## organizationRole

**Response:** Returns an OrganizationRole

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `Int!` |  |

```graphql
query organizationRole($id: Int!) {
  organizationRole(id: $id) {
    nodeId
    id
    name
    displayName
    description
    isSystem
    createdTimestamp
    createdUserId
    createdClientId
    lastModifiedTimestamp
    lastModifiedUserId
    lastModifiedClientId
    isDeleted
    level
    createdUser {
      ...UserFragment
    }
    lastModifiedUser {
      ...UserFragment
    }
    organizationUsers {
      ...OrganizationUsersConnectionFragment
    }
  }
}
```

## organizationRoleByName

**Response:** Returns an OrganizationRole

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `name` | `String!` |  |

```graphql
query organizationRoleByName($name: String!) {
  organizationRoleByName(name: $name) {
    nodeId
    id
    name
    displayName
    description
    isSystem
    createdTimestamp
    createdUserId
    createdClientId
    lastModifiedTimestamp
    lastModifiedUserId
    lastModifiedClientId
    isDeleted
    level
    createdUser {
      ...UserFragment
    }
    lastModifiedUser {
      ...UserFragment
    }
    organizationUsers {
      ...OrganizationUsersConnectionFragment
    }
  }
}
```

## organizationRoleByNodeId

Description Reads a single OrganizationRole using its globally unique ID .

**Response:** Returns an OrganizationRole

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |

```graphql
query organizationRoleByNodeId($nodeId: ID!) {
  organizationRoleByNodeId(nodeId: $nodeId) {
    nodeId
    id
    name
    displayName
    description
    isSystem
    createdTimestamp
    createdUserId
    createdClientId
    lastModifiedTimestamp
    lastModifiedUserId
    lastModifiedClientId
    isDeleted
    level
    createdUser {
      ...UserFragment
    }
    lastModifiedUser {
      ...UserFragment
    }
    organizationUsers {
      ...OrganizationUsersConnectionFragment
    }
  }
}
```

## organizationRoles

Description Reads and enables pagination through a set of OrganizationRole .

**Response:** Returns an OrganizationRolesConnection

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `first` | `Int` |  |
| `last` | `Int` |  |
| `offset` | `Int` |  |
| `before` | `Cursor` |  |
| `after` | `Cursor` |  |
| `orderBy` | `[OrganizationRolesOrderBy!]` |  |
| `condition` | `OrganizationRoleCondition` |  |
| `includeDeleted` | `IncludeDeletedOption` |  |
| `filter` | `OrganizationRoleFilter` |  |

```graphql
query organizationRoles(
  $first: Int,
  $last: Int,
  $offset: Int,
  $before: Cursor,
  $after: Cursor,
  $orderBy: [OrganizationRolesOrderBy!],
  $condition: OrganizationRoleCondition,
  $includeDeleted: IncludeDeletedOption,
  $filter: OrganizationRoleFilter
) {
  organizationRoles(
    first: $first,
    last: $last,
    offset: $offset,
    before: $before,
    after: $after,
    orderBy: $orderBy,
    condition: $condition,
    includeDeleted: $includeDeleted,
    filter: $filter
  ) {
    nodes {
      ...OrganizationRoleFragment
    }
    totalCount
  }
}
```

## organizationUser

**Response:** Returns an OrganizationUser

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `Int!` |  |

```graphql
query organizationUser($id: Int!) {
  organizationUser(id: $id) {
    nodeId
    id
    organizationId
    userId
    organizationRoleId
    createdTimestamp
    createdUserId
    createdClientId
    lastModifiedTimestamp
    lastModifiedUserId
    lastModifiedClientId
    isDeleted
    isEnabled
    organization {
      ...OrganizationFragment
    }
    user {
      ...UserFragment
    }
    organizationRole {
      ...OrganizationRoleFragment
    }
    createdUser {
      ...UserFragment
    }
    lastModifiedUser {
      ...UserFragment
    }
  }
}
```

## organizationUserByNodeId

Description Reads a single OrganizationUser using its globally unique ID .

**Response:** Returns an OrganizationUser

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |

```graphql
query organizationUserByNodeId($nodeId: ID!) {
  organizationUserByNodeId(nodeId: $nodeId) {
    nodeId
    id
    organizationId
    userId
    organizationRoleId
    createdTimestamp
    createdUserId
    createdClientId
    lastModifiedTimestamp
    lastModifiedUserId
    lastModifiedClientId
    isDeleted
    isEnabled
    organization {
      ...OrganizationFragment
    }
    user {
      ...UserFragment
    }
    organizationRole {
      ...OrganizationRoleFragment
    }
    createdUser {
      ...UserFragment
    }
    lastModifiedUser {
      ...UserFragment
    }
  }
}
```

## organizationUserByOrganizationIdAndUserId

**Response:** Returns an OrganizationUser

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `organizationId` | `Int!` |  |
| `userId` | `Int!` |  |

```graphql
query organizationUserByOrganizationIdAndUserId(
  $organizationId: Int!,
  $userId: Int!
) {
  organizationUserByOrganizationIdAndUserId(
    organizationId: $organizationId,
    userId: $userId
  ) {
    nodeId
    id
    organizationId
    userId
    organizationRoleId
    createdTimestamp
    createdUserId
    createdClientId
    lastModifiedTimestamp
    lastModifiedUserId
    lastModifiedClientId
    isDeleted
    isEnabled
    organization {
      ...OrganizationFragment
    }
    user {
      ...UserFragment
    }
    organizationRole {
      ...OrganizationRoleFragment
    }
    createdUser {
      ...UserFragment
    }
    lastModifiedUser {
      ...UserFragment
    }
  }
}
```

## organizationUsers

Description Reads and enables pagination through a set of OrganizationUser .

**Response:** Returns an OrganizationUsersConnection

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `first` | `Int` |  |
| `last` | `Int` |  |
| `offset` | `Int` |  |
| `before` | `Cursor` |  |
| `after` | `Cursor` |  |
| `orderBy` | `[OrganizationUsersOrderBy!]` |  |
| `condition` | `OrganizationUserCondition` |  |
| `includeDeleted` | `IncludeDeletedOption` |  |
| `filter` | `OrganizationUserFilter` |  |

```graphql
query organizationUsers(
  $first: Int,
  $last: Int,
  $offset: Int,
  $before: Cursor,
  $after: Cursor,
  $orderBy: [OrganizationUsersOrderBy!],
  $condition: OrganizationUserCondition,
  $includeDeleted: IncludeDeletedOption,
  $filter: OrganizationUserFilter
) {
  organizationUsers(
    first: $first,
    last: $last,
    offset: $offset,
    before: $before,
    after: $after,
    orderBy: $orderBy,
    condition: $condition,
    includeDeleted: $includeDeleted,
    filter: $filter
  ) {
    nodes {
      ...OrganizationUserFragment
    }
    totalCount
  }
}
```

## organizations

Description Reads and enables pagination through a set of Organization .

**Response:** Returns an OrganizationsConnection

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `first` | `Int` |  |
| `last` | `Int` |  |
| `offset` | `Int` |  |
| `before` | `Cursor` |  |
| `after` | `Cursor` |  |
| `orderBy` | `[OrganizationsOrderBy!]` |  |
| `condition` | `OrganizationCondition` |  |
| `includeDeleted` | `IncludeDeletedOption` |  |
| `filter` | `OrganizationFilter` |  |

```graphql
query organizations(
  $first: Int,
  $last: Int,
  $offset: Int,
  $before: Cursor,
  $after: Cursor,
  $orderBy: [OrganizationsOrderBy!],
  $condition: OrganizationCondition,
  $includeDeleted: IncludeDeletedOption,
  $filter: OrganizationFilter
) {
  organizations(
    first: $first,
    last: $last,
    offset: $offset,
    before: $before,
    after: $after,
    orderBy: $orderBy,
    condition: $condition,
    includeDeleted: $includeDeleted,
    filter: $filter
  ) {
    nodes {
      ...OrganizationFragment
    }
    totalCount
  }
}
```

## parkingAttribute

**Response:** Returns a ParkingAttribute

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `Int!` |  |

```graphql
query parkingAttribute($id: Int!) {
  parkingAttribute(id: $id) {
    nodeId
    id
    siteId
    name
    displayName
    isSystem
    ledRgbaValue
    displayRgbaValue
    reportRgbaValue
    isEnabled
    createdTimestamp
    createdUserId
    createdClientId
    lastModifiedTimestamp
    lastModifiedUserId
    lastModifiedClientId
    isDeleted
    priority
    remoteId
    site {
      ...SiteFragment
    }
    createdUser {
      ...UserFragment
    }
    lastModifiedUser {
      ...UserFragment
    }
    parkingZoneDataPoints {
      ...ParkingZoneDataPointsConnectionFragment
    }
    siteLevelParkingAttributeUsageByHours {
      ...SiteLevelParkingAttributeUsageByHoursConnectionFragment
    }
    parkingSpaceDataPoints {
      ...ParkingSpaceDataPointsConnectionFragment
    }
    latestSiteLevelParkingAttributeParkingUsages {
      ...LatestSiteLevelParkingAttributeParkingUsagesConnectionFragment
    }
    latestSiteParkingAttributeParkingUsages {
      ...LatestSiteParkingAttributeParkingUsagesConnectionFragment
    }
    reportingSiteLevelParkingAttributeUsageByHours {
      ...ReportingSiteLevelParkingAttributeUsageByHoursConnectionFragment
    }
  }
}
```

## parkingAttributeByNodeId

Description Reads a single ParkingAttribute using its globally unique ID .

**Response:** Returns a ParkingAttribute

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |

```graphql
query parkingAttributeByNodeId($nodeId: ID!) {
  parkingAttributeByNodeId(nodeId: $nodeId) {
    nodeId
    id
    siteId
    name
    displayName
    isSystem
    ledRgbaValue
    displayRgbaValue
    reportRgbaValue
    isEnabled
    createdTimestamp
    createdUserId
    createdClientId
    lastModifiedTimestamp
    lastModifiedUserId
    lastModifiedClientId
    isDeleted
    priority
    remoteId
    site {
      ...SiteFragment
    }
    createdUser {
      ...UserFragment
    }
    lastModifiedUser {
      ...UserFragment
    }
    parkingZoneDataPoints {
      ...ParkingZoneDataPointsConnectionFragment
    }
    siteLevelParkingAttributeUsageByHours {
      ...SiteLevelParkingAttributeUsageByHoursConnectionFragment
    }
    parkingSpaceDataPoints {
      ...ParkingSpaceDataPointsConnectionFragment
    }
    latestSiteLevelParkingAttributeParkingUsages {
      ...LatestSiteLevelParkingAttributeParkingUsagesConnectionFragment
    }
    latestSiteParkingAttributeParkingUsages {
      ...LatestSiteParkingAttributeParkingUsagesConnectionFragment
    }
    reportingSiteLevelParkingAttributeUsageByHours {
      ...ReportingSiteLevelParkingAttributeUsageByHoursConnectionFragment
    }
  }
}
```

## parkingAttributes

Description Reads and enables pagination through a set of ParkingAttribute .

**Response:** Returns a ParkingAttributesConnection

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `first` | `Int` |  |
| `last` | `Int` |  |
| `offset` | `Int` |  |
| `before` | `Cursor` |  |
| `after` | `Cursor` |  |
| `orderBy` | `[ParkingAttributesOrderBy!]` |  |
| `condition` | `ParkingAttributeCondition` |  |
| `includeDeleted` | `IncludeDeletedOption` |  |
| `filter` | `ParkingAttributeFilter` |  |

```graphql
query parkingAttributes(
  $first: Int,
  $last: Int,
  $offset: Int,
  $before: Cursor,
  $after: Cursor,
  $orderBy: [ParkingAttributesOrderBy!],
  $condition: ParkingAttributeCondition,
  $includeDeleted: IncludeDeletedOption,
  $filter: ParkingAttributeFilter
) {
  parkingAttributes(
    first: $first,
    last: $last,
    offset: $offset,
    before: $before,
    after: $after,
    orderBy: $orderBy,
    condition: $condition,
    includeDeleted: $includeDeleted,
    filter: $filter
  ) {
    nodes {
      ...ParkingAttributeFragment
    }
    totalCount
  }
}
```

## parkingSpace

**Response:** Returns a ParkingSpace

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `Int!` |  |

```graphql
query parkingSpace($id: Int!) {
  parkingSpace(id: $id) {
    nodeId
    id
    siteId
    siteLevelId
    name
    displayName
    description
    createdTimestamp
    createdUserId
    createdClientId
    lastModifiedTimestamp
    lastModifiedUserId
    lastModifiedClientId
    isDeleted
    indicatedBySensorIdOld
    detectedBySensorIdOld
    isEnabled
    indicatedBySensorId
    detectedBySensorId
    site {
      ...SiteFragment
    }
    siteLevel {
      ...SiteLevelFragment
    }
    createdUser {
      ...UserFragment
    }
    lastModifiedUser {
      ...UserFragment
    }
    indicatedBySensor {
      ...SensorFragment
    }
    detectedBySensor {
      ...SensorFragment
    }
    parkingSpaceDataPoints {
      ...ParkingSpaceDataPointsConnectionFragment
    }
    parkingSpaceVehicleSessions {
      ...ParkingSpaceVehicleSessionsConnectionFragment
    }
  }
}
```

## parkingSpaceByNodeId

Description Reads a single ParkingSpace using its globally unique ID .

**Response:** Returns a ParkingSpace

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |

```graphql
query parkingSpaceByNodeId($nodeId: ID!) {
  parkingSpaceByNodeId(nodeId: $nodeId) {
    nodeId
    id
    siteId
    siteLevelId
    name
    displayName
    description
    createdTimestamp
    createdUserId
    createdClientId
    lastModifiedTimestamp
    lastModifiedUserId
    lastModifiedClientId
    isDeleted
    indicatedBySensorIdOld
    detectedBySensorIdOld
    isEnabled
    indicatedBySensorId
    detectedBySensorId
    site {
      ...SiteFragment
    }
    siteLevel {
      ...SiteLevelFragment
    }
    createdUser {
      ...UserFragment
    }
    lastModifiedUser {
      ...UserFragment
    }
    indicatedBySensor {
      ...SensorFragment
    }
    detectedBySensor {
      ...SensorFragment
    }
    parkingSpaceDataPoints {
      ...ParkingSpaceDataPointsConnectionFragment
    }
    parkingSpaceVehicleSessions {
      ...ParkingSpaceVehicleSessionsConnectionFragment
    }
  }
}
```

## parkingSpaceDataPoint

**Response:** Returns a ParkingSpaceDataPoint

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `Int!` |  |
| `id` | `BigInt!` |  |

```graphql
query parkingSpaceDataPoint(
  $siteId: Int!,
  $id: BigInt!
) {
  parkingSpaceDataPoint(
    siteId: $siteId,
    id: $id
  ) {
    nodeId
    id
    parkingSpaceId
    siteLevelId
    parkingAttributeId
    siteId
    occupancyStatus
    period {
      ...DatetimeRangeFragment
    }
    duration
    parkingSpaceUtilizationEventRemoteId
    occupancyPeriod {
      ...DatetimeRangeFragment
    }
    occupancyDuration
    occupancyStartParkingSpaceDataPointId
    parkingSpace {
      ...ParkingSpaceFragment
    }
    siteLevel {
      ...SiteLevelFragment
    }
    parkingAttribute {
      ...ParkingAttributeFragment
    }
    site {
      ...SiteFragment
    }
    siteOccupancyStartParkingSpaceDataPoint {
      ...ParkingSpaceDataPointFragment
    }
    parkingSpaceDataPointsBySiteIdAndOccupancyStartParkingSpaceDataPointId {
      ...ParkingSpaceDataPointsConnectionFragment
    }
  }
}
```

## parkingSpaceDataPointByNodeId

Description Reads a single ParkingSpaceDataPoint using its globally unique ID .

**Response:** Returns a ParkingSpaceDataPoint

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |

```graphql
query parkingSpaceDataPointByNodeId($nodeId: ID!) {
  parkingSpaceDataPointByNodeId(nodeId: $nodeId) {
    nodeId
    id
    parkingSpaceId
    siteLevelId
    parkingAttributeId
    siteId
    occupancyStatus
    period {
      ...DatetimeRangeFragment
    }
    duration
    parkingSpaceUtilizationEventRemoteId
    occupancyPeriod {
      ...DatetimeRangeFragment
    }
    occupancyDuration
    occupancyStartParkingSpaceDataPointId
    parkingSpace {
      ...ParkingSpaceFragment
    }
    siteLevel {
      ...SiteLevelFragment
    }
    parkingAttribute {
      ...ParkingAttributeFragment
    }
    site {
      ...SiteFragment
    }
    siteOccupancyStartParkingSpaceDataPoint {
      ...ParkingSpaceDataPointFragment
    }
    parkingSpaceDataPointsBySiteIdAndOccupancyStartParkingSpaceDataPointId {
      ...ParkingSpaceDataPointsConnectionFragment
    }
  }
}
```

## parkingSpaceDataPoints

Description Reads and enables pagination through a set of ParkingSpaceDataPoint .

**Response:** Returns a ParkingSpaceDataPointsConnection

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `first` | `Int` |  |
| `last` | `Int` |  |
| `offset` | `Int` |  |
| `before` | `Cursor` |  |
| `after` | `Cursor` |  |
| `orderBy` | `[ParkingSpaceDataPointsOrderBy!]` |  |
| `condition` | `ParkingSpaceDataPointCondition` |  |
| `filter` | `ParkingSpaceDataPointFilter` |  |

```graphql
query parkingSpaceDataPoints(
  $first: Int,
  $last: Int,
  $offset: Int,
  $before: Cursor,
  $after: Cursor,
  $orderBy: [ParkingSpaceDataPointsOrderBy!],
  $condition: ParkingSpaceDataPointCondition,
  $filter: ParkingSpaceDataPointFilter
) {
  parkingSpaceDataPoints(
    first: $first,
    last: $last,
    offset: $offset,
    before: $before,
    after: $after,
    orderBy: $orderBy,
    condition: $condition,
    filter: $filter
  ) {
    nodes {
      ...ParkingSpaceDataPointFragment
    }
    totalCount
  }
}
```

## parkingSpaceVehicleSession

**Response:** Returns a ParkingSpaceVehicleSession

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `Int!` |  |
| `id` | `BigInt!` |  |

```graphql
query parkingSpaceVehicleSession(
  $siteId: Int!,
  $id: BigInt!
) {
  parkingSpaceVehicleSession(
    siteId: $siteId,
    id: $id
  ) {
    nodeId
    id
    remoteId
    parkingSpaceId
    isDeleted
    remoteUpdatedAt
    siteId
    period {
      ...DatetimeRangeFragment
    }
    duration
    modified
    siteVehicleSessionId
    siteVehicleSessionMatchConfidence
    remoteSiteVehicleSessionId
    remoteChangeSeqId
    updatedAt
    parkingSpace {
      ...ParkingSpaceFragment
    }
    site {
      ...SiteFragment
    }
    parkingSpaceVehicleSessionVehicleRecognitions {
      ...ParkingSpaceVehicleSessionVehicleRecognitionsConnectionFragment
    }
  }
}
```

## parkingSpaceVehicleSessionByNodeId

Description Reads a single ParkingSpaceVehicleSession using its globally unique ID .

**Response:** Returns a ParkingSpaceVehicleSession

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |

```graphql
query parkingSpaceVehicleSessionByNodeId($nodeId: ID!) {
  parkingSpaceVehicleSessionByNodeId(nodeId: $nodeId) {
    nodeId
    id
    remoteId
    parkingSpaceId
    isDeleted
    remoteUpdatedAt
    siteId
    period {
      ...DatetimeRangeFragment
    }
    duration
    modified
    siteVehicleSessionId
    siteVehicleSessionMatchConfidence
    remoteSiteVehicleSessionId
    remoteChangeSeqId
    updatedAt
    parkingSpace {
      ...ParkingSpaceFragment
    }
    site {
      ...SiteFragment
    }
    parkingSpaceVehicleSessionVehicleRecognitions {
      ...ParkingSpaceVehicleSessionVehicleRecognitionsConnectionFragment
    }
  }
}
```

## parkingSpaceVehicleSessionByRemoteIdAndSiteId

**Response:** Returns a ParkingSpaceVehicleSession

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `remoteId` | `Int!` |  |
| `siteId` | `Int!` |  |

```graphql
query parkingSpaceVehicleSessionByRemoteIdAndSiteId(
  $remoteId: Int!,
  $siteId: Int!
) {
  parkingSpaceVehicleSessionByRemoteIdAndSiteId(
    remoteId: $remoteId,
    siteId: $siteId
  ) {
    nodeId
    id
    remoteId
    parkingSpaceId
    isDeleted
    remoteUpdatedAt
    siteId
    period {
      ...DatetimeRangeFragment
    }
    duration
    modified
    siteVehicleSessionId
    siteVehicleSessionMatchConfidence
    remoteSiteVehicleSessionId
    remoteChangeSeqId
    updatedAt
    parkingSpace {
      ...ParkingSpaceFragment
    }
    site {
      ...SiteFragment
    }
    parkingSpaceVehicleSessionVehicleRecognitions {
      ...ParkingSpaceVehicleSessionVehicleRecognitionsConnectionFragment
    }
  }
}
```

## parkingSpaceVehicleSessionVehicleRecognition

**Response:** Returns a ParkingSpaceVehicleSessionVehicleRecognition

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `Int!` |  |
| `id` | `BigInt!` |  |

```graphql
query parkingSpaceVehicleSessionVehicleRecognition(
  $siteId: Int!,
  $id: BigInt!
) {
  parkingSpaceVehicleSessionVehicleRecognition(
    siteId: $siteId,
    id: $id
  ) {
    nodeId
    id
    remoteId
    parkingSpaceVehicleSessionId
    vehicleRecognitionId
    remoteUpdatedAt
    siteId
    remoteChangeSeqId
    updatedAt
    site {
      ...SiteFragment
    }
    parkingSpaceVehicleSession {
      ...ParkingSpaceVehicleSessionFragment
    }
    vehicleRecognition {
      ...VehicleRecognitionFragment
    }
  }
}
```

## parkingSpaceVehicleSessionVehicleRecognitionByNodeId

Description Reads a single ParkingSpaceVehicleSessionVehicleRecognition using its globally unique ID .

**Response:** Returns a ParkingSpaceVehicleSessionVehicleRecognition

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |

```graphql
query parkingSpaceVehicleSessionVehicleRecognitionByNodeId($nodeId: ID!) {
  parkingSpaceVehicleSessionVehicleRecognitionByNodeId(nodeId: $nodeId) {
    nodeId
    id
    remoteId
    parkingSpaceVehicleSessionId
    vehicleRecognitionId
    remoteUpdatedAt
    siteId
    remoteChangeSeqId
    updatedAt
    site {
      ...SiteFragment
    }
    parkingSpaceVehicleSession {
      ...ParkingSpaceVehicleSessionFragment
    }
    vehicleRecognition {
      ...VehicleRecognitionFragment
    }
  }
}
```

## parkingSpaceVehicleSessionVehicleRecognitionByRemoteIdAndSiteId

**Response:** Returns a ParkingSpaceVehicleSessionVehicleRecognition

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `remoteId` | `Int!` |  |
| `siteId` | `Int!` |  |

```graphql
query parkingSpaceVehicleSessionVehicleRecognitionByRemoteIdAndSiteId(
  $remoteId: Int!,
  $siteId: Int!
) {
  parkingSpaceVehicleSessionVehicleRecognitionByRemoteIdAndSiteId(
    remoteId: $remoteId,
    siteId: $siteId
  ) {
    nodeId
    id
    remoteId
    parkingSpaceVehicleSessionId
    vehicleRecognitionId
    remoteUpdatedAt
    siteId
    remoteChangeSeqId
    updatedAt
    site {
      ...SiteFragment
    }
    parkingSpaceVehicleSession {
      ...ParkingSpaceVehicleSessionFragment
    }
    vehicleRecognition {
      ...VehicleRecognitionFragment
    }
  }
}
```

## parkingSpaceVehicleSessionVehicleRecognitions

Description Reads and enables pagination through a set of ParkingSpaceVehicleSessionVehicleRecognition .

**Response:** Returns a ParkingSpaceVehicleSessionVehicleRecognitionsConnection

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `first` | `Int` |  |
| `last` | `Int` |  |
| `offset` | `Int` |  |
| `before` | `Cursor` |  |
| `after` | `Cursor` |  |
| `orderBy` | `[ParkingSpaceVehicleSessionVehicleRecognitionsOrderBy!]` |  |
| `condition` | `ParkingSpaceVehicleSessionVehicleRecognitionCondition` |  |
| `filter` | `ParkingSpaceVehicleSessionVehicleRecognitionFilter` |  |

```graphql
query parkingSpaceVehicleSessionVehicleRecognitions(
  $first: Int,
  $last: Int,
  $offset: Int,
  $before: Cursor,
  $after: Cursor,
  $orderBy: [ParkingSpaceVehicleSessionVehicleRecognitionsOrderBy!],
  $condition: ParkingSpaceVehicleSessionVehicleRecognitionCondition,
  $filter: ParkingSpaceVehicleSessionVehicleRecognitionFilter
) {
  parkingSpaceVehicleSessionVehicleRecognitions(
    first: $first,
    last: $last,
    offset: $offset,
    before: $before,
    after: $after,
    orderBy: $orderBy,
    condition: $condition,
    filter: $filter
  ) {
    nodes {
      ...ParkingSpaceVehicleSessionVehicleRecognitionFragment
    }
    totalCount
  }
}
```

## parkingSpaceVehicleSessions

Description Reads and enables pagination through a set of ParkingSpaceVehicleSession .

**Response:** Returns a ParkingSpaceVehicleSessionsConnection

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `first` | `Int` |  |
| `last` | `Int` |  |
| `offset` | `Int` |  |
| `before` | `Cursor` |  |
| `after` | `Cursor` |  |
| `orderBy` | `[ParkingSpaceVehicleSessionsOrderBy!]` |  |
| `condition` | `ParkingSpaceVehicleSessionCondition` |  |
| `includeDeleted` | `IncludeDeletedOption` |  |
| `filter` | `ParkingSpaceVehicleSessionFilter` |  |

```graphql
query parkingSpaceVehicleSessions(
  $first: Int,
  $last: Int,
  $offset: Int,
  $before: Cursor,
  $after: Cursor,
  $orderBy: [ParkingSpaceVehicleSessionsOrderBy!],
  $condition: ParkingSpaceVehicleSessionCondition,
  $includeDeleted: IncludeDeletedOption,
  $filter: ParkingSpaceVehicleSessionFilter
) {
  parkingSpaceVehicleSessions(
    first: $first,
    last: $last,
    offset: $offset,
    before: $before,
    after: $after,
    orderBy: $orderBy,
    condition: $condition,
    includeDeleted: $includeDeleted,
    filter: $filter
  ) {
    nodes {
      ...ParkingSpaceVehicleSessionFragment
    }
    totalCount
  }
}
```

## parkingSpaces

Description Reads and enables pagination through a set of ParkingSpace .

**Response:** Returns a ParkingSpacesConnection

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `first` | `Int` |  |
| `last` | `Int` |  |
| `offset` | `Int` |  |
| `before` | `Cursor` |  |
| `after` | `Cursor` |  |
| `orderBy` | `[ParkingSpacesOrderBy!]` |  |
| `condition` | `ParkingSpaceCondition` |  |
| `includeDeleted` | `IncludeDeletedOption` |  |
| `filter` | `ParkingSpaceFilter` |  |

```graphql
query parkingSpaces(
  $first: Int,
  $last: Int,
  $offset: Int,
  $before: Cursor,
  $after: Cursor,
  $orderBy: [ParkingSpacesOrderBy!],
  $condition: ParkingSpaceCondition,
  $includeDeleted: IncludeDeletedOption,
  $filter: ParkingSpaceFilter
) {
  parkingSpaces(
    first: $first,
    last: $last,
    offset: $offset,
    before: $before,
    after: $after,
    orderBy: $orderBy,
    condition: $condition,
    includeDeleted: $includeDeleted,
    filter: $filter
  ) {
    nodes {
      ...ParkingSpaceFragment
    }
    totalCount
  }
}
```

## parkingZone

**Response:** Returns a ParkingZone

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `Int!` |  |

```graphql
query parkingZone($id: Int!) {
  parkingZone(id: $id) {
    nodeId
    id
    siteId
    siteLevelId
    name
    displayName
    description
    createdTimestamp
    createdUserId
    createdClientId
    lastModifiedTimestamp
    lastModifiedUserId
    lastModifiedClientId
    isDeleted
    indicatedBySensorIdOld
    remoteId
    countedBySensorId
    indicatedBySensorId
    site {
      ...SiteFragment
    }
    siteLevel {
      ...SiteLevelFragment
    }
    createdUser {
      ...UserFragment
    }
    lastModifiedUser {
      ...UserFragment
    }
    countedBySensor {
      ...SensorFragment
    }
    indicatedBySensor {
      ...SensorFragment
    }
    parkingZoneDataPoints {
      ...ParkingZoneDataPointsConnectionFragment
    }
    parkingZoneCounters {
      ...ParkingZoneCountersConnectionFragment
    }
  }
}
```

## parkingZoneByNodeId

Description Reads a single ParkingZone using its globally unique ID .

**Response:** Returns a ParkingZone

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |

```graphql
query parkingZoneByNodeId($nodeId: ID!) {
  parkingZoneByNodeId(nodeId: $nodeId) {
    nodeId
    id
    siteId
    siteLevelId
    name
    displayName
    description
    createdTimestamp
    createdUserId
    createdClientId
    lastModifiedTimestamp
    lastModifiedUserId
    lastModifiedClientId
    isDeleted
    indicatedBySensorIdOld
    remoteId
    countedBySensorId
    indicatedBySensorId
    site {
      ...SiteFragment
    }
    siteLevel {
      ...SiteLevelFragment
    }
    createdUser {
      ...UserFragment
    }
    lastModifiedUser {
      ...UserFragment
    }
    countedBySensor {
      ...SensorFragment
    }
    indicatedBySensor {
      ...SensorFragment
    }
    parkingZoneDataPoints {
      ...ParkingZoneDataPointsConnectionFragment
    }
    parkingZoneCounters {
      ...ParkingZoneCountersConnectionFragment
    }
  }
}
```

## parkingZoneCounter

**Response:** Returns a ParkingZoneCounter

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `Int!` |  |

```graphql
query parkingZoneCounter($id: Int!) {
  parkingZoneCounter(id: $id) {
    nodeId
    id
    entered
    left
    lastReset
    beginTimestamp
    endTimestamp
    parkingZoneId
    siteId
    parkingZone {
      ...ParkingZoneFragment
    }
    site {
      ...SiteFragment
    }
  }
}
```

## parkingZoneCounterByNodeId

Description Reads a single ParkingZoneCounter using its globally unique ID .

**Response:** Returns a ParkingZoneCounter

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |

```graphql
query parkingZoneCounterByNodeId($nodeId: ID!) {
  parkingZoneCounterByNodeId(nodeId: $nodeId) {
    nodeId
    id
    entered
    left
    lastReset
    beginTimestamp
    endTimestamp
    parkingZoneId
    siteId
    parkingZone {
      ...ParkingZoneFragment
    }
    site {
      ...SiteFragment
    }
  }
}
```

## parkingZoneCounters

Description Reads and enables pagination through a set of ParkingZoneCounter .

**Response:** Returns a ParkingZoneCountersConnection

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `first` | `Int` |  |
| `last` | `Int` |  |
| `offset` | `Int` |  |
| `before` | `Cursor` |  |
| `after` | `Cursor` |  |
| `orderBy` | `[ParkingZoneCountersOrderBy!]` |  |
| `condition` | `ParkingZoneCounterCondition` |  |
| `filter` | `ParkingZoneCounterFilter` |  |

```graphql
query parkingZoneCounters(
  $first: Int,
  $last: Int,
  $offset: Int,
  $before: Cursor,
  $after: Cursor,
  $orderBy: [ParkingZoneCountersOrderBy!],
  $condition: ParkingZoneCounterCondition,
  $filter: ParkingZoneCounterFilter
) {
  parkingZoneCounters(
    first: $first,
    last: $last,
    offset: $offset,
    before: $before,
    after: $after,
    orderBy: $orderBy,
    condition: $condition,
    filter: $filter
  ) {
    nodes {
      ...ParkingZoneCounterFragment
    }
    totalCount
  }
}
```

## parkingZoneDataPoint

**Response:** Returns a ParkingZoneDataPoint

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigInt!` |  |

```graphql
query parkingZoneDataPoint($id: BigInt!) {
  parkingZoneDataPoint(id: $id) {
    nodeId
    id
    parkingZoneId
    siteLevelId
    parkingAttributeId
    siteId
    availableCount
    totalCount
    period {
      ...DatetimeRangeFragment
    }
    parkingZoneUtilizationEventRemoteId
    availableCountDelta
    parkingZone {
      ...ParkingZoneFragment
    }
    siteLevel {
      ...SiteLevelFragment
    }
    parkingAttribute {
      ...ParkingAttributeFragment
    }
    site {
      ...SiteFragment
    }
  }
}
```

## parkingZoneDataPointByNodeId

Description Reads a single ParkingZoneDataPoint using its globally unique ID .

**Response:** Returns a ParkingZoneDataPoint

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |

```graphql
query parkingZoneDataPointByNodeId($nodeId: ID!) {
  parkingZoneDataPointByNodeId(nodeId: $nodeId) {
    nodeId
    id
    parkingZoneId
    siteLevelId
    parkingAttributeId
    siteId
    availableCount
    totalCount
    period {
      ...DatetimeRangeFragment
    }
    parkingZoneUtilizationEventRemoteId
    availableCountDelta
    parkingZone {
      ...ParkingZoneFragment
    }
    siteLevel {
      ...SiteLevelFragment
    }
    parkingAttribute {
      ...ParkingAttributeFragment
    }
    site {
      ...SiteFragment
    }
  }
}
```

## parkingZoneDataPoints

Description Reads and enables pagination through a set of ParkingZoneDataPoint .

**Response:** Returns a ParkingZoneDataPointsConnection

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `first` | `Int` |  |
| `last` | `Int` |  |
| `offset` | `Int` |  |
| `before` | `Cursor` |  |
| `after` | `Cursor` |  |
| `orderBy` | `[ParkingZoneDataPointsOrderBy!]` |  |
| `condition` | `ParkingZoneDataPointCondition` |  |
| `filter` | `ParkingZoneDataPointFilter` |  |

```graphql
query parkingZoneDataPoints(
  $first: Int,
  $last: Int,
  $offset: Int,
  $before: Cursor,
  $after: Cursor,
  $orderBy: [ParkingZoneDataPointsOrderBy!],
  $condition: ParkingZoneDataPointCondition,
  $filter: ParkingZoneDataPointFilter
) {
  parkingZoneDataPoints(
    first: $first,
    last: $last,
    offset: $offset,
    before: $before,
    after: $after,
    orderBy: $orderBy,
    condition: $condition,
    filter: $filter
  ) {
    nodes {
      ...ParkingZoneDataPointFragment
    }
    totalCount
  }
}
```

## parkingZones

Description Reads and enables pagination through a set of ParkingZone .

**Response:** Returns a ParkingZonesConnection

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `first` | `Int` |  |
| `last` | `Int` |  |
| `offset` | `Int` |  |
| `before` | `Cursor` |  |
| `after` | `Cursor` |  |
| `orderBy` | `[ParkingZonesOrderBy!]` |  |
| `condition` | `ParkingZoneCondition` |  |
| `includeDeleted` | `IncludeDeletedOption` |  |
| `filter` | `ParkingZoneFilter` |  |

```graphql
query parkingZones(
  $first: Int,
  $last: Int,
  $offset: Int,
  $before: Cursor,
  $after: Cursor,
  $orderBy: [ParkingZonesOrderBy!],
  $condition: ParkingZoneCondition,
  $includeDeleted: IncludeDeletedOption,
  $filter: ParkingZoneFilter
) {
  parkingZones(
    first: $first,
    last: $last,
    offset: $offset,
    before: $before,
    after: $after,
    orderBy: $orderBy,
    condition: $condition,
    includeDeleted: $includeDeleted,
    filter: $filter
  ) {
    nodes {
      ...ParkingZoneFragment
    }
    totalCount
  }
}
```

## query

Description Exposes the root query type nested one level down. This is helpful for Relay 1 which can only query top level fields if they are in a particular form.

**Response:** Returns a Query!

```graphql
query query {
  query {
    query {
      ...QueryFragment
    }
    nodeId
    node {
      ...NodeFragment
    }
    users {
      ...UsersConnectionFragment
    }
    latestSiteLevelParkingAttributeParkingUsages {
      ...LatestSiteLevelParkingAttributeParkingUsagesConnectionFragment
    }
    latestSiteParkingAttributeParkingUsages {
      ...LatestSiteParkingAttributeParkingUsagesConnectionFragment
    }
    latestSiteParkingUsages {
      ...LatestSiteParkingUsagesConnectionFragment
    }
    organizations {
      ...OrganizationsConnectionFragment
    }
    organizationRoles {
      ...OrganizationRolesConnectionFragment
    }
    organizationUsers {
      ...OrganizationUsersConnectionFragment
    }
    parkingAttributes {
      ...ParkingAttributesConnectionFragment
    }
    parkingSpaces {
      ...ParkingSpacesConnectionFragment
    }
    parkingSpaceDataPoints {
      ...ParkingSpaceDataPointsConnectionFragment
    }
    parkingSpaceVehicleSessions {
      ...ParkingSpaceVehicleSessionsConnectionFragment
    }
    parkingSpaceVehicleSessionVehicleRecognitions {
      ...ParkingSpaceVehicleSessionVehicleRecognitionsConnectionFragment
    }
    parkingZones {
      ...ParkingZonesConnectionFragment
    }
    parkingZoneCounters {
      ...ParkingZoneCountersConnectionFragment
    }
    parkingZoneDataPoints {
      ...ParkingZoneDataPointsConnectionFragment
    }
    reportingSiteLevelParkingAttributeUsageByHours {
      ...ReportingSiteLevelParkingAttributeUsageByHoursConnectionFragment
    }
    sensors {
      ...SensorsConnectionFragment
    }
    sites {
      ...SitesConnectionFragment
    }
    siteLevels {
      ...SiteLevelsConnectionFragment
    }
    siteLevelParkingAttributeUsageByHours {
      ...SiteLevelParkingAttributeUsageByHoursConnectionFragment
    }
    vehicleRecognitions {
      ...VehicleRecognitionsConnectionFragment
    }
    vehicleRecognitionPlates {
      ...VehicleRecognitionPlatesConnectionFragment
    }
    user {
      ...UserFragment
    }
    userBySubjectId {
      ...UserFragment
    }
    organization {
      ...OrganizationFragment
    }
    organizationRole {
      ...OrganizationRoleFragment
    }
    organizationRoleByName {
      ...OrganizationRoleFragment
    }
    organizationUser {
      ...OrganizationUserFragment
    }
    organizationUserByOrganizationIdAndUserId {
      ...OrganizationUserFragment
    }
    parkingAttribute {
      ...ParkingAttributeFragment
    }
    parkingSpace {
      ...ParkingSpaceFragment
    }
    parkingSpaceDataPoint {
      ...ParkingSpaceDataPointFragment
    }
    parkingSpaceVehicleSessionByRemoteIdAndSiteId {
      ...ParkingSpaceVehicleSessionFragment
    }
    parkingSpaceVehicleSession {
      ...ParkingSpaceVehicleSessionFragment
    }
    parkingSpaceVehicleSessionVehicleRecognitionByRemoteIdAndSiteId {
      ...ParkingSpaceVehicleSessionVehicleRecognitionFragment
    }
    parkingSpaceVehicleSessionVehicleRecognition {
      ...ParkingSpaceVehicleSessionVehicleRecognitionFragment
    }
    parkingZone {
      ...ParkingZoneFragment
    }
    parkingZoneCounter {
      ...ParkingZoneCounterFragment
    }
    parkingZoneDataPoint {
      ...ParkingZoneDataPointFragment
    }
    sensor {
      ...SensorFragment
    }
    sensorBySensorIdAndSiteId {
      ...SensorFragment
    }
    site {
      ...SiteFragment
    }
    siteLevel {
      ...SiteLevelFragment
    }
    siteLevelParkingAttributeUsageByHour {
      ...SiteLevelParkingAttributeUsageByHourFragment
    }
    vehicleRecognitionByRemoteIdAndSiteId {
      ...VehicleRecognitionFragment
    }
    vehicleRecognition {
      ...VehicleRecognitionFragment
    }
    vehicleRecognitionPlateByRemoteIdAndSiteId {
      ...VehicleRecognitionPlateFragment
    }
    vehicleRecognitionPlate {
      ...VehicleRecognitionPlateFragment
    }
    currentClientId
    currentUser {
      ...UserFragment
    }
    currentUserId
    currentUserOrganizationRole
    userByNodeId {
      ...UserFragment
    }
    organizationByNodeId {
      ...OrganizationFragment
    }
    organizationRoleByNodeId {
      ...OrganizationRoleFragment
    }
    organizationUserByNodeId {
      ...OrganizationUserFragment
    }
    parkingAttributeByNodeId {
      ...ParkingAttributeFragment
    }
    parkingSpaceByNodeId {
      ...ParkingSpaceFragment
    }
    parkingSpaceDataPointByNodeId {
      ...ParkingSpaceDataPointFragment
    }
    parkingSpaceVehicleSessionByNodeId {
      ...ParkingSpaceVehicleSessionFragment
    }
    parkingSpaceVehicleSessionVehicleRecognitionByNodeId {
      ...ParkingSpaceVehicleSessionVehicleRecognitionFragment
    }
    parkingZoneByNodeId {
      ...ParkingZoneFragment
    }
    parkingZoneCounterByNodeId {
      ...ParkingZoneCounterFragment
    }
    parkingZoneDataPointByNodeId {
      ...ParkingZoneDataPointFragment
    }
    sensorByNodeId {
      ...SensorFragment
    }
    siteByNodeId {
      ...SiteFragment
    }
    siteLevelByNodeId {
      ...SiteLevelFragment
    }
    siteLevelParkingAttributeUsageByHourByNodeId {
      ...SiteLevelParkingAttributeUsageByHourFragment
    }
    vehicleRecognitionByNodeId {
      ...VehicleRecognitionFragment
    }
    vehicleRecognitionPlateByNodeId {
      ...VehicleRecognitionPlateFragment
    }
  }
}
```

## reportingSiteLevelParkingAttributeUsageByHours

Description Reads and enables pagination through a set of ReportingSiteLevelParkingAttributeUsageByHour .

**Response:** Returns a ReportingSiteLevelParkingAttributeUsageByHoursConnection

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `first` | `Int` |  |
| `last` | `Int` |  |
| `offset` | `Int` |  |
| `before` | `Cursor` |  |
| `after` | `Cursor` |  |
| `orderBy` | `[ReportingSiteLevelParkingAttributeUsageByHoursOrderBy!]` |  |
| `condition` | `ReportingSiteLevelParkingAttributeUsageByHourCondition` |  |
| `filter` | `ReportingSiteLevelParkingAttributeUsageByHourFilter` |  |

```graphql
query reportingSiteLevelParkingAttributeUsageByHours(
  $first: Int,
  $last: Int,
  $offset: Int,
  $before: Cursor,
  $after: Cursor,
  $orderBy: [ReportingSiteLevelParkingAttributeUsageByHoursOrderBy!],
  $condition: ReportingSiteLevelParkingAttributeUsageByHourCondition,
  $filter: ReportingSiteLevelParkingAttributeUsageByHourFilter
) {
  reportingSiteLevelParkingAttributeUsageByHours(
    first: $first,
    last: $last,
    offset: $offset,
    before: $before,
    after: $after,
    orderBy: $orderBy,
    condition: $condition,
    filter: $filter
  ) {
    nodes {
      ...ReportingSiteLevelParkingAttributeUsageByHourFragment
    }
    totalCount
  }
}
```

## sensor

**Response:** Returns a Sensor

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigInt!` |  |

```graphql
query sensor($id: BigInt!) {
  sensor(id: $id) {
    nodeId
    id
    sensorId
    configurationName
    configurationDescription
    siteId
    isDeleted
    site {
      ...SiteFragment
    }
    parkingSpacesByIndicatedBySensor {
      ...ParkingSpacesConnectionFragment
    }
    parkingSpacesByDetectedBySensor {
      ...ParkingSpacesConnectionFragment
    }
    parkingZonesByCountedBySensor {
      ...ParkingZonesConnectionFragment
    }
    parkingZonesByIndicatedBySensor {
      ...ParkingZonesConnectionFragment
    }
  }
}
```

## sensorByNodeId

Description Reads a single Sensor using its globally unique ID .

**Response:** Returns a Sensor

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |

```graphql
query sensorByNodeId($nodeId: ID!) {
  sensorByNodeId(nodeId: $nodeId) {
    nodeId
    id
    sensorId
    configurationName
    configurationDescription
    siteId
    isDeleted
    site {
      ...SiteFragment
    }
    parkingSpacesByIndicatedBySensor {
      ...ParkingSpacesConnectionFragment
    }
    parkingSpacesByDetectedBySensor {
      ...ParkingSpacesConnectionFragment
    }
    parkingZonesByCountedBySensor {
      ...ParkingZonesConnectionFragment
    }
    parkingZonesByIndicatedBySensor {
      ...ParkingZonesConnectionFragment
    }
  }
}
```

## sensorBySensorIdAndSiteId

**Response:** Returns a Sensor

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `sensorId` | `String!` |  |
| `siteId` | `Int!` |  |

```graphql
query sensorBySensorIdAndSiteId(
  $sensorId: String!,
  $siteId: Int!
) {
  sensorBySensorIdAndSiteId(
    sensorId: $sensorId,
    siteId: $siteId
  ) {
    nodeId
    id
    sensorId
    configurationName
    configurationDescription
    siteId
    isDeleted
    site {
      ...SiteFragment
    }
    parkingSpacesByIndicatedBySensor {
      ...ParkingSpacesConnectionFragment
    }
    parkingSpacesByDetectedBySensor {
      ...ParkingSpacesConnectionFragment
    }
    parkingZonesByCountedBySensor {
      ...ParkingZonesConnectionFragment
    }
    parkingZonesByIndicatedBySensor {
      ...ParkingZonesConnectionFragment
    }
  }
}
```

## sensors

Description Reads and enables pagination through a set of Sensor .

**Response:** Returns a SensorsConnection

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `first` | `Int` |  |
| `last` | `Int` |  |
| `offset` | `Int` |  |
| `before` | `Cursor` |  |
| `after` | `Cursor` |  |
| `orderBy` | `[SensorsOrderBy!]` |  |
| `condition` | `SensorCondition` |  |
| `includeDeleted` | `IncludeDeletedOption` |  |
| `filter` | `SensorFilter` |  |

```graphql
query sensors(
  $first: Int,
  $last: Int,
  $offset: Int,
  $before: Cursor,
  $after: Cursor,
  $orderBy: [SensorsOrderBy!],
  $condition: SensorCondition,
  $includeDeleted: IncludeDeletedOption,
  $filter: SensorFilter
) {
  sensors(
    first: $first,
    last: $last,
    offset: $offset,
    before: $before,
    after: $after,
    orderBy: $orderBy,
    condition: $condition,
    includeDeleted: $includeDeleted,
    filter: $filter
  ) {
    nodes {
      ...SensorFragment
    }
    totalCount
  }
}
```

## site

**Response:** Returns a Site

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `Int!` |  |

```graphql
query site($id: Int!) {
  site(id: $id) {
    nodeId
    id
    siteUuid
    name
    displayName
    description
    address1
    address2
    address3
    directions
    timeZoneName
    organizationId
    isDataPollingActive
    createdTimestamp
    createdUserId
    createdClientId
    lastModifiedTimestamp
    lastModifiedUserId
    lastModifiedClientId
    isDeleted
    indicatorLightsEnabled
    perimeterVehicleTrackingEnabled
    vaultId
    parkingSpaceVehicleTrackingEnabled
    guidanceDisabledLedRgbaValue
    guidanceDisabledDisplayRgbaValue
    guidanceUnavailableLedRgbaValue
    guidanceUnavailableDisplayRgbaValue
    organization {
      ...OrganizationFragment
    }
    createdUser {
      ...UserFragment
    }
    lastModifiedUser {
      ...UserFragment
    }
    parkingZoneDataPoints {
      ...ParkingZoneDataPointsConnectionFragment
    }
    siteLevelParkingAttributeUsageByHours {
      ...SiteLevelParkingAttributeUsageByHoursConnectionFragment
    }
    parkingZoneCounters {
      ...ParkingZoneCountersConnectionFragment
    }
    parkingAttributes {
      ...ParkingAttributesConnectionFragment
    }
    parkingSpaces {
      ...ParkingSpacesConnectionFragment
    }
    parkingZones {
      ...ParkingZonesConnectionFragment
    }
    siteLevels {
      ...SiteLevelsConnectionFragment
    }
    sensors {
      ...SensorsConnectionFragment
    }
    parkingSpaceDataPoints {
      ...ParkingSpaceDataPointsConnectionFragment
    }
    vehicleRecognitions {
      ...VehicleRecognitionsConnectionFragment
    }
    vehicleRecognitionPlates {
      ...VehicleRecognitionPlatesConnectionFragment
    }
    parkingSpaceVehicleSessions {
      ...ParkingSpaceVehicleSessionsConnectionFragment
    }
    parkingSpaceVehicleSessionVehicleRecognitions {
      ...ParkingSpaceVehicleSessionVehicleRecognitionsConnectionFragment
    }
    latestSiteLevelParkingAttributeParkingUsages {
      ...LatestSiteLevelParkingAttributeParkingUsagesConnectionFragment
    }
    latestSiteParkingAttributeParkingUsages {
      ...LatestSiteParkingAttributeParkingUsagesConnectionFragment
    }
    latestSiteParkingUsages {
      ...LatestSiteParkingUsagesConnectionFragment
    }
    reportingSiteLevelParkingAttributeUsageByHours {
      ...ReportingSiteLevelParkingAttributeUsageByHoursConnectionFragment
    }
  }
}
```

## siteByNodeId

Description Reads a single Site using its globally unique ID .

**Response:** Returns a Site

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |

```graphql
query siteByNodeId($nodeId: ID!) {
  siteByNodeId(nodeId: $nodeId) {
    nodeId
    id
    siteUuid
    name
    displayName
    description
    address1
    address2
    address3
    directions
    timeZoneName
    organizationId
    isDataPollingActive
    createdTimestamp
    createdUserId
    createdClientId
    lastModifiedTimestamp
    lastModifiedUserId
    lastModifiedClientId
    isDeleted
    indicatorLightsEnabled
    perimeterVehicleTrackingEnabled
    vaultId
    parkingSpaceVehicleTrackingEnabled
    guidanceDisabledLedRgbaValue
    guidanceDisabledDisplayRgbaValue
    guidanceUnavailableLedRgbaValue
    guidanceUnavailableDisplayRgbaValue
    organization {
      ...OrganizationFragment
    }
    createdUser {
      ...UserFragment
    }
    lastModifiedUser {
      ...UserFragment
    }
    parkingZoneDataPoints {
      ...ParkingZoneDataPointsConnectionFragment
    }
    siteLevelParkingAttributeUsageByHours {
      ...SiteLevelParkingAttributeUsageByHoursConnectionFragment
    }
    parkingZoneCounters {
      ...ParkingZoneCountersConnectionFragment
    }
    parkingAttributes {
      ...ParkingAttributesConnectionFragment
    }
    parkingSpaces {
      ...ParkingSpacesConnectionFragment
    }
    parkingZones {
      ...ParkingZonesConnectionFragment
    }
    siteLevels {
      ...SiteLevelsConnectionFragment
    }
    sensors {
      ...SensorsConnectionFragment
    }
    parkingSpaceDataPoints {
      ...ParkingSpaceDataPointsConnectionFragment
    }
    vehicleRecognitions {
      ...VehicleRecognitionsConnectionFragment
    }
    vehicleRecognitionPlates {
      ...VehicleRecognitionPlatesConnectionFragment
    }
    parkingSpaceVehicleSessions {
      ...ParkingSpaceVehicleSessionsConnectionFragment
    }
    parkingSpaceVehicleSessionVehicleRecognitions {
      ...ParkingSpaceVehicleSessionVehicleRecognitionsConnectionFragment
    }
    latestSiteLevelParkingAttributeParkingUsages {
      ...LatestSiteLevelParkingAttributeParkingUsagesConnectionFragment
    }
    latestSiteParkingAttributeParkingUsages {
      ...LatestSiteParkingAttributeParkingUsagesConnectionFragment
    }
    latestSiteParkingUsages {
      ...LatestSiteParkingUsagesConnectionFragment
    }
    reportingSiteLevelParkingAttributeUsageByHours {
      ...ReportingSiteLevelParkingAttributeUsageByHoursConnectionFragment
    }
  }
}
```

## siteLevel

**Response:** Returns a SiteLevel

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `Int!` |  |

```graphql
query siteLevel($id: Int!) {
  siteLevel(id: $id) {
    nodeId
    id
    siteId
    position
    displayName
    createdTimestamp
    createdUserId
    createdClientId
    lastModifiedTimestamp
    lastModifiedUserId
    lastModifiedClientId
    isDeleted
    mapFilePath
    site {
      ...SiteFragment
    }
    createdUser {
      ...UserFragment
    }
    lastModifiedUser {
      ...UserFragment
    }
    parkingZoneDataPoints {
      ...ParkingZoneDataPointsConnectionFragment
    }
    siteLevelParkingAttributeUsageByHours {
      ...SiteLevelParkingAttributeUsageByHoursConnectionFragment
    }
    parkingSpaces {
      ...ParkingSpacesConnectionFragment
    }
    parkingZones {
      ...ParkingZonesConnectionFragment
    }
    parkingSpaceDataPoints {
      ...ParkingSpaceDataPointsConnectionFragment
    }
    latestSiteLevelParkingAttributeParkingUsages {
      ...LatestSiteLevelParkingAttributeParkingUsagesConnectionFragment
    }
    reportingSiteLevelParkingAttributeUsageByHours {
      ...ReportingSiteLevelParkingAttributeUsageByHoursConnectionFragment
    }
    mapFilePathSigned
  }
}
```

## siteLevelByNodeId

Description Reads a single SiteLevel using its globally unique ID .

**Response:** Returns a SiteLevel

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |

```graphql
query siteLevelByNodeId($nodeId: ID!) {
  siteLevelByNodeId(nodeId: $nodeId) {
    nodeId
    id
    siteId
    position
    displayName
    createdTimestamp
    createdUserId
    createdClientId
    lastModifiedTimestamp
    lastModifiedUserId
    lastModifiedClientId
    isDeleted
    mapFilePath
    site {
      ...SiteFragment
    }
    createdUser {
      ...UserFragment
    }
    lastModifiedUser {
      ...UserFragment
    }
    parkingZoneDataPoints {
      ...ParkingZoneDataPointsConnectionFragment
    }
    siteLevelParkingAttributeUsageByHours {
      ...SiteLevelParkingAttributeUsageByHoursConnectionFragment
    }
    parkingSpaces {
      ...ParkingSpacesConnectionFragment
    }
    parkingZones {
      ...ParkingZonesConnectionFragment
    }
    parkingSpaceDataPoints {
      ...ParkingSpaceDataPointsConnectionFragment
    }
    latestSiteLevelParkingAttributeParkingUsages {
      ...LatestSiteLevelParkingAttributeParkingUsagesConnectionFragment
    }
    reportingSiteLevelParkingAttributeUsageByHours {
      ...ReportingSiteLevelParkingAttributeUsageByHoursConnectionFragment
    }
    mapFilePathSigned
  }
}
```

## siteLevelParkingAttributeUsageByHour

**Response:** Returns a SiteLevelParkingAttributeUsageByHour

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `hour` | `Datetime!` |  |
| `siteLevelId` | `Int!` |  |
| `parkingAttributeId` | `Int!` |  |

```graphql
query siteLevelParkingAttributeUsageByHour(
  $hour: Datetime!,
  $siteLevelId: Int!,
  $parkingAttributeId: Int!
) {
  siteLevelParkingAttributeUsageByHour(
    hour: $hour,
    siteLevelId: $siteLevelId,
    parkingAttributeId: $parkingAttributeId
  ) {
    nodeId
    siteId
    siteLevelId
    parkingAttributeId
    hour
    availableCount
    noDataCount
    occupiedCount
    totalCount
    finalizedAt
    site {
      ...SiteFragment
    }
    siteLevel {
      ...SiteLevelFragment
    }
    parkingAttribute {
      ...ParkingAttributeFragment
    }
  }
}
```

## siteLevelParkingAttributeUsageByHourByNodeId

Description Reads a single SiteLevelParkingAttributeUsageByHour using its globally unique ID .

**Response:** Returns a SiteLevelParkingAttributeUsageByHour

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |

```graphql
query siteLevelParkingAttributeUsageByHourByNodeId($nodeId: ID!) {
  siteLevelParkingAttributeUsageByHourByNodeId(nodeId: $nodeId) {
    nodeId
    siteId
    siteLevelId
    parkingAttributeId
    hour
    availableCount
    noDataCount
    occupiedCount
    totalCount
    finalizedAt
    site {
      ...SiteFragment
    }
    siteLevel {
      ...SiteLevelFragment
    }
    parkingAttribute {
      ...ParkingAttributeFragment
    }
  }
}
```

## siteLevelParkingAttributeUsageByHours

Description Reads and enables pagination through a set of SiteLevelParkingAttributeUsageByHour .

**Response:** Returns a SiteLevelParkingAttributeUsageByHoursConnection

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `first` | `Int` |  |
| `last` | `Int` |  |
| `offset` | `Int` |  |
| `before` | `Cursor` |  |
| `after` | `Cursor` |  |
| `orderBy` | `[SiteLevelParkingAttributeUsageByHoursOrderBy!]` |  |
| `condition` | `SiteLevelParkingAttributeUsageByHourCondition` |  |
| `filter` | `SiteLevelParkingAttributeUsageByHourFilter` |  |

```graphql
query siteLevelParkingAttributeUsageByHours(
  $first: Int,
  $last: Int,
  $offset: Int,
  $before: Cursor,
  $after: Cursor,
  $orderBy: [SiteLevelParkingAttributeUsageByHoursOrderBy!],
  $condition: SiteLevelParkingAttributeUsageByHourCondition,
  $filter: SiteLevelParkingAttributeUsageByHourFilter
) {
  siteLevelParkingAttributeUsageByHours(
    first: $first,
    last: $last,
    offset: $offset,
    before: $before,
    after: $after,
    orderBy: $orderBy,
    condition: $condition,
    filter: $filter
  ) {
    nodes {
      ...SiteLevelParkingAttributeUsageByHourFragment
    }
    totalCount
  }
}
```

## siteLevels

Description Reads and enables pagination through a set of SiteLevel .

**Response:** Returns a SiteLevelsConnection

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `first` | `Int` |  |
| `last` | `Int` |  |
| `offset` | `Int` |  |
| `before` | `Cursor` |  |
| `after` | `Cursor` |  |
| `orderBy` | `[SiteLevelsOrderBy!]` |  |
| `condition` | `SiteLevelCondition` |  |
| `includeDeleted` | `IncludeDeletedOption` |  |
| `filter` | `SiteLevelFilter` |  |

```graphql
query siteLevels(
  $first: Int,
  $last: Int,
  $offset: Int,
  $before: Cursor,
  $after: Cursor,
  $orderBy: [SiteLevelsOrderBy!],
  $condition: SiteLevelCondition,
  $includeDeleted: IncludeDeletedOption,
  $filter: SiteLevelFilter
) {
  siteLevels(
    first: $first,
    last: $last,
    offset: $offset,
    before: $before,
    after: $after,
    orderBy: $orderBy,
    condition: $condition,
    includeDeleted: $includeDeleted,
    filter: $filter
  ) {
    nodes {
      ...SiteLevelFragment
    }
    totalCount
  }
}
```

## sites

Description Reads and enables pagination through a set of Site .

**Response:** Returns a SitesConnection

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `first` | `Int` |  |
| `last` | `Int` |  |
| `offset` | `Int` |  |
| `before` | `Cursor` |  |
| `after` | `Cursor` |  |
| `orderBy` | `[SitesOrderBy!]` |  |
| `condition` | `SiteCondition` |  |
| `includeDeleted` | `IncludeDeletedOption` |  |
| `filter` | `SiteFilter` |  |

```graphql
query sites(
  $first: Int,
  $last: Int,
  $offset: Int,
  $before: Cursor,
  $after: Cursor,
  $orderBy: [SitesOrderBy!],
  $condition: SiteCondition,
  $includeDeleted: IncludeDeletedOption,
  $filter: SiteFilter
) {
  sites(
    first: $first,
    last: $last,
    offset: $offset,
    before: $before,
    after: $after,
    orderBy: $orderBy,
    condition: $condition,
    includeDeleted: $includeDeleted,
    filter: $filter
  ) {
    nodes {
      ...SiteFragment
    }
    totalCount
  }
}
```

## user

**Response:** Returns a User

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `Int!` |  |

```graphql
query user($id: Int!) {
  user(id: $id) {
    nodeId
    id
    email
    givenName
    familyName
    createdTimestamp
    createdUserId
    createdClientId
    lastModifiedTimestamp
    lastModifiedUserId
    lastModifiedClientId
    subjectId
    isDeleted
    createdUser {
      ...UserFragment
    }
    lastModifiedUser {
      ...UserFragment
    }
    usersByCreatedUserId {
      ...UsersConnectionFragment
    }
    usersByLastModifiedUserId {
      ...UsersConnectionFragment
    }
    organizationsByCreatedUserId {
      ...OrganizationsConnectionFragment
    }
    organizationsByLastModifiedUserId {
      ...OrganizationsConnectionFragment
    }
    organizationUsersByUserId {
      ...OrganizationUsersConnectionFragment
    }
    organizationUsersByCreatedUserId {
      ...OrganizationUsersConnectionFragment
    }
    organizationUsersByLastModifiedUserId {
      ...OrganizationUsersConnectionFragment
    }
    parkingAttributesByCreatedUserId {
      ...ParkingAttributesConnectionFragment
    }
    parkingAttributesByLastModifiedUserId {
      ...ParkingAttributesConnectionFragment
    }
    parkingSpacesByCreatedUserId {
      ...ParkingSpacesConnectionFragment
    }
    parkingSpacesByLastModifiedUserId {
      ...ParkingSpacesConnectionFragment
    }
    parkingZonesByCreatedUserId {
      ...ParkingZonesConnectionFragment
    }
    parkingZonesByLastModifiedUserId {
      ...ParkingZonesConnectionFragment
    }
    sitesByCreatedUserId {
      ...SitesConnectionFragment
    }
    sitesByLastModifiedUserId {
      ...SitesConnectionFragment
    }
    siteLevelsByCreatedUserId {
      ...SiteLevelsConnectionFragment
    }
    siteLevelsByLastModifiedUserId {
      ...SiteLevelsConnectionFragment
    }
    organizationRolesByCreatedUserId {
      ...OrganizationRolesConnectionFragment
    }
    organizationRolesByLastModifiedUserId {
      ...OrganizationRolesConnectionFragment
    }
  }
}
```

## userByNodeId

Description Reads a single User using its globally unique ID .

**Response:** Returns a User

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |

```graphql
query userByNodeId($nodeId: ID!) {
  userByNodeId(nodeId: $nodeId) {
    nodeId
    id
    email
    givenName
    familyName
    createdTimestamp
    createdUserId
    createdClientId
    lastModifiedTimestamp
    lastModifiedUserId
    lastModifiedClientId
    subjectId
    isDeleted
    createdUser {
      ...UserFragment
    }
    lastModifiedUser {
      ...UserFragment
    }
    usersByCreatedUserId {
      ...UsersConnectionFragment
    }
    usersByLastModifiedUserId {
      ...UsersConnectionFragment
    }
    organizationsByCreatedUserId {
      ...OrganizationsConnectionFragment
    }
    organizationsByLastModifiedUserId {
      ...OrganizationsConnectionFragment
    }
    organizationUsersByUserId {
      ...OrganizationUsersConnectionFragment
    }
    organizationUsersByCreatedUserId {
      ...OrganizationUsersConnectionFragment
    }
    organizationUsersByLastModifiedUserId {
      ...OrganizationUsersConnectionFragment
    }
    parkingAttributesByCreatedUserId {
      ...ParkingAttributesConnectionFragment
    }
    parkingAttributesByLastModifiedUserId {
      ...ParkingAttributesConnectionFragment
    }
    parkingSpacesByCreatedUserId {
      ...ParkingSpacesConnectionFragment
    }
    parkingSpacesByLastModifiedUserId {
      ...ParkingSpacesConnectionFragment
    }
    parkingZonesByCreatedUserId {
      ...ParkingZonesConnectionFragment
    }
    parkingZonesByLastModifiedUserId {
      ...ParkingZonesConnectionFragment
    }
    sitesByCreatedUserId {
      ...SitesConnectionFragment
    }
    sitesByLastModifiedUserId {
      ...SitesConnectionFragment
    }
    siteLevelsByCreatedUserId {
      ...SiteLevelsConnectionFragment
    }
    siteLevelsByLastModifiedUserId {
      ...SiteLevelsConnectionFragment
    }
    organizationRolesByCreatedUserId {
      ...OrganizationRolesConnectionFragment
    }
    organizationRolesByLastModifiedUserId {
      ...OrganizationRolesConnectionFragment
    }
  }
}
```

## userBySubjectId

**Response:** Returns a User

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `subjectId` | `UUID!` |  |

```graphql
query userBySubjectId($subjectId: UUID!) {
  userBySubjectId(subjectId: $subjectId) {
    nodeId
    id
    email
    givenName
    familyName
    createdTimestamp
    createdUserId
    createdClientId
    lastModifiedTimestamp
    lastModifiedUserId
    lastModifiedClientId
    subjectId
    isDeleted
    createdUser {
      ...UserFragment
    }
    lastModifiedUser {
      ...UserFragment
    }
    usersByCreatedUserId {
      ...UsersConnectionFragment
    }
    usersByLastModifiedUserId {
      ...UsersConnectionFragment
    }
    organizationsByCreatedUserId {
      ...OrganizationsConnectionFragment
    }
    organizationsByLastModifiedUserId {
      ...OrganizationsConnectionFragment
    }
    organizationUsersByUserId {
      ...OrganizationUsersConnectionFragment
    }
    organizationUsersByCreatedUserId {
      ...OrganizationUsersConnectionFragment
    }
    organizationUsersByLastModifiedUserId {
      ...OrganizationUsersConnectionFragment
    }
    parkingAttributesByCreatedUserId {
      ...ParkingAttributesConnectionFragment
    }
    parkingAttributesByLastModifiedUserId {
      ...ParkingAttributesConnectionFragment
    }
    parkingSpacesByCreatedUserId {
      ...ParkingSpacesConnectionFragment
    }
    parkingSpacesByLastModifiedUserId {
      ...ParkingSpacesConnectionFragment
    }
    parkingZonesByCreatedUserId {
      ...ParkingZonesConnectionFragment
    }
    parkingZonesByLastModifiedUserId {
      ...ParkingZonesConnectionFragment
    }
    sitesByCreatedUserId {
      ...SitesConnectionFragment
    }
    sitesByLastModifiedUserId {
      ...SitesConnectionFragment
    }
    siteLevelsByCreatedUserId {
      ...SiteLevelsConnectionFragment
    }
    siteLevelsByLastModifiedUserId {
      ...SiteLevelsConnectionFragment
    }
    organizationRolesByCreatedUserId {
      ...OrganizationRolesConnectionFragment
    }
    organizationRolesByLastModifiedUserId {
      ...OrganizationRolesConnectionFragment
    }
  }
}
```

## users

Description Reads and enables pagination through a set of User .

**Response:** Returns a UsersConnection

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `first` | `Int` |  |
| `last` | `Int` |  |
| `offset` | `Int` |  |
| `before` | `Cursor` |  |
| `after` | `Cursor` |  |
| `orderBy` | `[UsersOrderBy!]` |  |
| `condition` | `UserCondition` |  |
| `includeDeleted` | `IncludeDeletedOption` |  |
| `filter` | `UserFilter` |  |

```graphql
query users(
  $first: Int,
  $last: Int,
  $offset: Int,
  $before: Cursor,
  $after: Cursor,
  $orderBy: [UsersOrderBy!],
  $condition: UserCondition,
  $includeDeleted: IncludeDeletedOption,
  $filter: UserFilter
) {
  users(
    first: $first,
    last: $last,
    offset: $offset,
    before: $before,
    after: $after,
    orderBy: $orderBy,
    condition: $condition,
    includeDeleted: $includeDeleted,
    filter: $filter
  ) {
    nodes {
      ...UserFragment
    }
    totalCount
  }
}
```

## vehicleRecognition

**Response:** Returns a VehicleRecognition

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `Int!` |  |
| `id` | `BigInt!` |  |

```graphql
query vehicleRecognition(
  $siteId: Int!,
  $id: BigInt!
) {
  vehicleRecognition(
    siteId: $siteId,
    id: $id
  ) {
    nodeId
    id
    remoteId
    timestamp
    dscore
    location
    siteId
    cameraId
    remoteImagePath
    imageUrl
    remoteUpdatedAt
    remoteMakeModelImagePath
    makeModelImageUrl
    plateBoxXMin
    plateBoxXMax
    plateBoxYMin
    plateBoxYMax
    imageWidth
    imageHeight
    compositeImageUrl
    remoteChangeSeqId
    updatedAt
    site {
      ...SiteFragment
    }
    vehicleRecognitionPlates {
      ...VehicleRecognitionPlatesConnectionFragment
    }
    parkingSpaceVehicleSessionVehicleRecognitions {
      ...ParkingSpaceVehicleSessionVehicleRecognitionsConnectionFragment
    }
    imageUrlSigned
    makeModelImageUrlSigned
    compositeImageUrlSigned
  }
}
```

## vehicleRecognitionByNodeId

Description Reads a single VehicleRecognition using its globally unique ID .

**Response:** Returns a VehicleRecognition

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |

```graphql
query vehicleRecognitionByNodeId($nodeId: ID!) {
  vehicleRecognitionByNodeId(nodeId: $nodeId) {
    nodeId
    id
    remoteId
    timestamp
    dscore
    location
    siteId
    cameraId
    remoteImagePath
    imageUrl
    remoteUpdatedAt
    remoteMakeModelImagePath
    makeModelImageUrl
    plateBoxXMin
    plateBoxXMax
    plateBoxYMin
    plateBoxYMax
    imageWidth
    imageHeight
    compositeImageUrl
    remoteChangeSeqId
    updatedAt
    site {
      ...SiteFragment
    }
    vehicleRecognitionPlates {
      ...VehicleRecognitionPlatesConnectionFragment
    }
    parkingSpaceVehicleSessionVehicleRecognitions {
      ...ParkingSpaceVehicleSessionVehicleRecognitionsConnectionFragment
    }
    imageUrlSigned
    makeModelImageUrlSigned
    compositeImageUrlSigned
  }
}
```

## vehicleRecognitionByRemoteIdAndSiteId

**Response:** Returns a VehicleRecognition

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `remoteId` | `Int!` |  |
| `siteId` | `Int!` |  |

```graphql
query vehicleRecognitionByRemoteIdAndSiteId(
  $remoteId: Int!,
  $siteId: Int!
) {
  vehicleRecognitionByRemoteIdAndSiteId(
    remoteId: $remoteId,
    siteId: $siteId
  ) {
    nodeId
    id
    remoteId
    timestamp
    dscore
    location
    siteId
    cameraId
    remoteImagePath
    imageUrl
    remoteUpdatedAt
    remoteMakeModelImagePath
    makeModelImageUrl
    plateBoxXMin
    plateBoxXMax
    plateBoxYMin
    plateBoxYMax
    imageWidth
    imageHeight
    compositeImageUrl
    remoteChangeSeqId
    updatedAt
    site {
      ...SiteFragment
    }
    vehicleRecognitionPlates {
      ...VehicleRecognitionPlatesConnectionFragment
    }
    parkingSpaceVehicleSessionVehicleRecognitions {
      ...ParkingSpaceVehicleSessionVehicleRecognitionsConnectionFragment
    }
    imageUrlSigned
    makeModelImageUrlSigned
    compositeImageUrlSigned
  }
}
```

## vehicleRecognitionPlate

**Response:** Returns a VehicleRecognitionPlate

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `Int!` |  |
| `id` | `BigInt!` |  |

```graphql
query vehicleRecognitionPlate(
  $siteId: Int!,
  $id: BigInt!
) {
  vehicleRecognitionPlate(
    siteId: $siteId,
    id: $id
  ) {
    nodeId
    id
    remoteId
    plate
    score
    primary
    vehicleRecognitionId
    siteId
    remoteUpdatedAt
    remoteChangeSeqId
    site {
      ...SiteFragment
    }
    vehicleRecognition {
      ...VehicleRecognitionFragment
    }
  }
}
```

## vehicleRecognitionPlateByNodeId

Description Reads a single VehicleRecognitionPlate using its globally unique ID .

**Response:** Returns a VehicleRecognitionPlate

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |

```graphql
query vehicleRecognitionPlateByNodeId($nodeId: ID!) {
  vehicleRecognitionPlateByNodeId(nodeId: $nodeId) {
    nodeId
    id
    remoteId
    plate
    score
    primary
    vehicleRecognitionId
    siteId
    remoteUpdatedAt
    remoteChangeSeqId
    site {
      ...SiteFragment
    }
    vehicleRecognition {
      ...VehicleRecognitionFragment
    }
  }
}
```

## vehicleRecognitionPlateByRemoteIdAndSiteId

**Response:** Returns a VehicleRecognitionPlate

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `remoteId` | `Int!` |  |
| `siteId` | `Int!` |  |

```graphql
query vehicleRecognitionPlateByRemoteIdAndSiteId(
  $remoteId: Int!,
  $siteId: Int!
) {
  vehicleRecognitionPlateByRemoteIdAndSiteId(
    remoteId: $remoteId,
    siteId: $siteId
  ) {
    nodeId
    id
    remoteId
    plate
    score
    primary
    vehicleRecognitionId
    siteId
    remoteUpdatedAt
    remoteChangeSeqId
    site {
      ...SiteFragment
    }
    vehicleRecognition {
      ...VehicleRecognitionFragment
    }
  }
}
```

## vehicleRecognitionPlates

Description Reads and enables pagination through a set of VehicleRecognitionPlate .

**Response:** Returns a VehicleRecognitionPlatesConnection

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `first` | `Int` |  |
| `last` | `Int` |  |
| `offset` | `Int` |  |
| `before` | `Cursor` |  |
| `after` | `Cursor` |  |
| `orderBy` | `[VehicleRecognitionPlatesOrderBy!]` |  |
| `condition` | `VehicleRecognitionPlateCondition` |  |
| `filter` | `VehicleRecognitionPlateFilter` |  |

```graphql
query vehicleRecognitionPlates(
  $first: Int,
  $last: Int,
  $offset: Int,
  $before: Cursor,
  $after: Cursor,
  $orderBy: [VehicleRecognitionPlatesOrderBy!],
  $condition: VehicleRecognitionPlateCondition,
  $filter: VehicleRecognitionPlateFilter
) {
  vehicleRecognitionPlates(
    first: $first,
    last: $last,
    offset: $offset,
    before: $before,
    after: $after,
    orderBy: $orderBy,
    condition: $condition,
    filter: $filter
  ) {
    nodes {
      ...VehicleRecognitionPlateFragment
    }
    totalCount
  }
}
```

## vehicleRecognitions

Description Reads and enables pagination through a set of VehicleRecognition .

**Response:** Returns a VehicleRecognitionsConnection

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `first` | `Int` |  |
| `last` | `Int` |  |
| `offset` | `Int` |  |
| `before` | `Cursor` |  |
| `after` | `Cursor` |  |
| `orderBy` | `[VehicleRecognitionsOrderBy!]` |  |
| `condition` | `VehicleRecognitionCondition` |  |
| `filter` | `VehicleRecognitionFilter` |  |

```graphql
query vehicleRecognitions(
  $first: Int,
  $last: Int,
  $offset: Int,
  $before: Cursor,
  $after: Cursor,
  $orderBy: [VehicleRecognitionsOrderBy!],
  $condition: VehicleRecognitionCondition,
  $filter: VehicleRecognitionFilter
) {
  vehicleRecognitions(
    first: $first,
    last: $last,
    offset: $offset,
    before: $before,
    after: $after,
    orderBy: $orderBy,
    condition: $condition,
    filter: $filter
  ) {
    nodes {
      ...VehicleRecognitionFragment
    }
    totalCount
  }
}
```

# Type Definitions

## BigFloat (scalar)

Description A floating point number that requires more precision than IEEE 754 binary 64

## BigFloatFilter (input-object)

Description A filter to be used against BigFloat fields. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `isNull` | `Boolean` |  |
| `equalTo` | `BigFloat` |  |
| `notEqualTo` | `BigFloat` |  |
| `distinctFrom` | `BigFloat` |  |
| `notDistinctFrom` | `BigFloat` |  |
| `in` | `[BigFloat!]` |  |
| `notIn` | `[BigFloat!]` |  |
| `lessThan` | `BigFloat` |  |
| `lessThanOrEqualTo` | `BigFloat` |  |
| `greaterThan` | `BigFloat` |  |
| `greaterThanOrEqualTo` | `BigFloat` |  |

## BigInt (scalar)

Description A signed eight-byte integer. The upper big integer values are greater than the max value for a JavaScript number. Therefore all big integers will be output as strings and not numbers.

## BigIntFilter (input-object)

Description A filter to be used against BigInt fields. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `isNull` | `Boolean` |  |
| `equalTo` | `BigInt` |  |
| `notEqualTo` | `BigInt` |  |
| `distinctFrom` | `BigInt` |  |
| `notDistinctFrom` | `BigInt` |  |
| `in` | `[BigInt!]` |  |
| `notIn` | `[BigInt!]` |  |
| `lessThan` | `BigInt` |  |
| `lessThanOrEqualTo` | `BigInt` |  |
| `greaterThan` | `BigInt` |  |
| `greaterThanOrEqualTo` | `BigInt` |  |

## Boolean (scalar)

Description The Boolean scalar type represents true or false .

## BooleanFilter (input-object)

Description A filter to be used against Boolean fields. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `isNull` | `Boolean` |  |
| `equalTo` | `Boolean` |  |
| `notEqualTo` | `Boolean` |  |
| `distinctFrom` | `Boolean` |  |
| `notDistinctFrom` | `Boolean` |  |
| `in` | `[Boolean!]` |  |
| `notIn` | `[Boolean!]` |  |
| `lessThan` | `Boolean` |  |
| `lessThanOrEqualTo` | `Boolean` |  |
| `greaterThan` | `Boolean` |  |
| `greaterThanOrEqualTo` | `Boolean` |  |

## Cursor (scalar)

Description A location in a connection that can be used for resuming pagination.

## Datetime (scalar)

Description A point in time as described by the ISO 8601 standard. May or may not include a timezone.

## DatetimeFilter (input-object)

Description A filter to be used against Datetime fields. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `isNull` | `Boolean` |  |
| `equalTo` | `Datetime` |  |
| `notEqualTo` | `Datetime` |  |
| `distinctFrom` | `Datetime` |  |
| `notDistinctFrom` | `Datetime` |  |
| `in` | `[Datetime!]` |  |
| `notIn` | `[Datetime!]` |  |
| `lessThan` | `Datetime` |  |
| `lessThanOrEqualTo` | `Datetime` |  |
| `greaterThan` | `Datetime` |  |
| `greaterThanOrEqualTo` | `Datetime` |  |

## DatetimeRange (object)

Description A range of Datetime .

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `start` | `DatetimeRangeBound` |  |
| `end` | `DatetimeRangeBound` |  |

## DatetimeRangeBound (object)

Description The value at one end of a range. A range can either include this value, or not.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `value` | `Datetime!` |  |
| `inclusive` | `Boolean!` |  |

## DatetimeRangeBoundInput (input-object)

Description The value at one end of a range. A range can either include this value, or not.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `value` | `Datetime!` |  |
| `inclusive` | `Boolean!` |  |

## DatetimeRangeFilter (input-object)

Description A filter to be used against DatetimeRange fields. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `isNull` | `Boolean` |  |
| `equalTo` | `DatetimeRangeInput` |  |
| `notEqualTo` | `DatetimeRangeInput` |  |
| `distinctFrom` | `DatetimeRangeInput` |  |
| `notDistinctFrom` | `DatetimeRangeInput` |  |
| `in` | `[DatetimeRangeInput!]` |  |
| `notIn` | `[DatetimeRangeInput!]` |  |
| `lessThan` | `DatetimeRangeInput` |  |
| `lessThanOrEqualTo` | `DatetimeRangeInput` |  |
| `greaterThan` | `DatetimeRangeInput` |  |
| `greaterThanOrEqualTo` | `DatetimeRangeInput` |  |
| `contains` | `DatetimeRangeInput` |  |
| `containsElement` | `Datetime` |  |
| `containedBy` | `DatetimeRangeInput` |  |
| `overlaps` | `DatetimeRangeInput` |  |
| `strictlyLeftOf` | `DatetimeRangeInput` |  |
| `strictlyRightOf` | `DatetimeRangeInput` |  |
| `notExtendsRightOf` | `DatetimeRangeInput` |  |
| `notExtendsLeftOf` | `DatetimeRangeInput` |  |
| `adjacentTo` | `DatetimeRangeInput` |  |
| `startUnbounded` | `Boolean` |  |
| `endUnbounded` | `Boolean` |  |

## DatetimeRangeInput (input-object)

Description A range of Datetime .

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `start` | `DatetimeRangeBoundInput` |  |
| `end` | `DatetimeRangeBoundInput` |  |

## HavingBigintFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `equalTo` | `BigInt` |  |
| `notEqualTo` | `BigInt` |  |
| `greaterThan` | `BigInt` |  |
| `greaterThanOrEqualTo` | `BigInt` |  |
| `lessThan` | `BigInt` |  |
| `lessThanOrEqualTo` | `BigInt` |  |

## HavingDatetimeFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `equalTo` | `Datetime` |  |
| `notEqualTo` | `Datetime` |  |
| `greaterThan` | `Datetime` |  |
| `greaterThanOrEqualTo` | `Datetime` |  |
| `lessThan` | `Datetime` |  |
| `lessThanOrEqualTo` | `Datetime` |  |

## HavingIntFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `equalTo` | `Int` |  |
| `notEqualTo` | `Int` |  |
| `greaterThan` | `Int` |  |
| `greaterThanOrEqualTo` | `Int` |  |
| `lessThan` | `Int` |  |
| `lessThanOrEqualTo` | `Int` |  |

## ID (scalar)

Description The ID scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as "4" ) or integer (such as 4 ) input value will be accepted as an ID.

## IncludeDeletedOption (enum)

Description Indicates whether deleted items should be included in the results or not.

## Int (scalar)

Description The Int scalar type represents non-fractional signed whole numeric values. Int can represent values between -(2^31) and 2^31 - 1.

## IntFilter (input-object)

Description A filter to be used against Int fields. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `isNull` | `Boolean` |  |
| `equalTo` | `Int` |  |
| `notEqualTo` | `Int` |  |
| `distinctFrom` | `Int` |  |
| `notDistinctFrom` | `Int` |  |
| `in` | `[Int!]` |  |
| `notIn` | `[Int!]` |  |
| `lessThan` | `Int` |  |
| `lessThanOrEqualTo` | `Int` |  |
| `greaterThan` | `Int` |  |
| `greaterThanOrEqualTo` | `Int` |  |

## LatestSiteLevelParkingAttributeParkingUsage (object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `Datetime` |  |
| `siteLevelId` | `Int` |  |
| `parkingAttributeId` | `Int` |  |
| `siteId` | `Int` |  |
| `availableCount` | `Int` |  |
| `noDataCount` | `Int` |  |
| `obscuredCount` | `Int` |  |
| `occupiedCount` | `Int` |  |
| `totalCount` | `Int` |  |
| `utilizedPercent` | `BigFloat` |  |
| `site` | `Site` |  |
| `siteLevel` | `SiteLevel` |  |
| `parkingAttribute` | `ParkingAttribute` |  |

## LatestSiteLevelParkingAttributeParkingUsageAggregatesFilter (input-object)

Description A filter to be used against aggregates of LatestSiteLevelParkingAttributeParkingUsage object types.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `sum` | `LatestSiteLevelParkingAttributeParkingUsageSumAggregateFilter` |  |
| `distinctCount` | `LatestSiteLevelParkingAttributeParkingUsageDistinctCountAggregateFilter` |  |
| `min` | `LatestSiteLevelParkingAttributeParkingUsageMinAggregateFilter` |  |
| `max` | `LatestSiteLevelParkingAttributeParkingUsageMaxAggregateFilter` |  |
| `average` | `LatestSiteLevelParkingAttributeParkingUsageAverageAggregateFilter` |  |
| `stddevSample` | `LatestSiteLevelParkingAttributeParkingUsageStddevSampleAggregateFilter` |  |
| `stddevPopulation` | `LatestSiteLevelParkingAttributeParkingUsageStddevPopulationAggregateFilter` |  |
| `varianceSample` | `LatestSiteLevelParkingAttributeParkingUsageVarianceSampleAggregateFilter` |  |
| `variancePopulation` | `LatestSiteLevelParkingAttributeParkingUsageVariancePopulationAggregateFilter` |  |

## LatestSiteLevelParkingAttributeParkingUsageAverageAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `obscuredCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |

## LatestSiteLevelParkingAttributeParkingUsageCondition (input-object)

Description A condition to be used against LatestSiteLevelParkingAttributeParkingUsage object types. All fields are tested for equality and combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `Datetime` |  |
| `siteLevelId` | `Int` |  |
| `parkingAttributeId` | `Int` |  |
| `siteId` | `Int` |  |
| `availableCount` | `Int` |  |
| `noDataCount` | `Int` |  |
| `obscuredCount` | `Int` |  |
| `occupiedCount` | `Int` |  |
| `totalCount` | `Int` |  |
| `utilizedPercent` | `BigFloat` |  |

## LatestSiteLevelParkingAttributeParkingUsageDistinctCountAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `BigIntFilter` |  |
| `siteLevelId` | `BigIntFilter` |  |
| `parkingAttributeId` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `availableCount` | `BigIntFilter` |  |
| `noDataCount` | `BigIntFilter` |  |
| `obscuredCount` | `BigIntFilter` |  |
| `occupiedCount` | `BigIntFilter` |  |
| `totalCount` | `BigIntFilter` |  |
| `utilizedPercent` | `BigIntFilter` |  |

## LatestSiteLevelParkingAttributeParkingUsageFilter (input-object)

Description A filter to be used against LatestSiteLevelParkingAttributeParkingUsage object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DatetimeFilter` |  |
| `siteLevelId` | `IntFilter` |  |
| `parkingAttributeId` | `IntFilter` |  |
| `siteId` | `IntFilter` |  |
| `availableCount` | `IntFilter` |  |
| `noDataCount` | `IntFilter` |  |
| `obscuredCount` | `IntFilter` |  |
| `occupiedCount` | `IntFilter` |  |
| `totalCount` | `IntFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |
| `siteExists` | `Boolean` |  |
| `siteLevelExists` | `Boolean` |  |
| `parkingAttributeExists` | `Boolean` |  |

## LatestSiteLevelParkingAttributeParkingUsageHavingAverageInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteLevelParkingAttributeParkingUsageHavingDistinctCountInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteLevelParkingAttributeParkingUsageHavingInput (input-object)

Description Conditions for LatestSiteLevelParkingAttributeParkingUsage aggregates.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `AND` | `[LatestSiteLevelParkingAttributeParkingUsageHavingInput!]` |  |
| `OR` | `[LatestSiteLevelParkingAttributeParkingUsageHavingInput!]` |  |
| `sum` | `LatestSiteLevelParkingAttributeParkingUsageHavingSumInput` |  |
| `distinctCount` | `LatestSiteLevelParkingAttributeParkingUsageHavingDistinctCountInput` |  |
| `min` | `LatestSiteLevelParkingAttributeParkingUsageHavingMinInput` |  |
| `max` | `LatestSiteLevelParkingAttributeParkingUsageHavingMaxInput` |  |
| `average` | `LatestSiteLevelParkingAttributeParkingUsageHavingAverageInput` |  |
| `stddevSample` | `LatestSiteLevelParkingAttributeParkingUsageHavingStddevSampleInput` |  |
| `stddevPopulation` | `LatestSiteLevelParkingAttributeParkingUsageHavingStddevPopulationInput` |  |
| `varianceSample` | `LatestSiteLevelParkingAttributeParkingUsageHavingVarianceSampleInput` |  |
| `variancePopulation` | `LatestSiteLevelParkingAttributeParkingUsageHavingVariancePopulationInput` |  |

## LatestSiteLevelParkingAttributeParkingUsageHavingMaxInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteLevelParkingAttributeParkingUsageHavingMinInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteLevelParkingAttributeParkingUsageHavingStddevPopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteLevelParkingAttributeParkingUsageHavingStddevSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteLevelParkingAttributeParkingUsageHavingSumInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteLevelParkingAttributeParkingUsageHavingVariancePopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteLevelParkingAttributeParkingUsageHavingVarianceSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteLevelParkingAttributeParkingUsageMaxAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DatetimeFilter` |  |
| `siteLevelId` | `IntFilter` |  |
| `parkingAttributeId` | `IntFilter` |  |
| `siteId` | `IntFilter` |  |
| `availableCount` | `IntFilter` |  |
| `noDataCount` | `IntFilter` |  |
| `obscuredCount` | `IntFilter` |  |
| `occupiedCount` | `IntFilter` |  |
| `totalCount` | `IntFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |

## LatestSiteLevelParkingAttributeParkingUsageMinAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DatetimeFilter` |  |
| `siteLevelId` | `IntFilter` |  |
| `parkingAttributeId` | `IntFilter` |  |
| `siteId` | `IntFilter` |  |
| `availableCount` | `IntFilter` |  |
| `noDataCount` | `IntFilter` |  |
| `obscuredCount` | `IntFilter` |  |
| `occupiedCount` | `IntFilter` |  |
| `totalCount` | `IntFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |

## LatestSiteLevelParkingAttributeParkingUsageStddevPopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `obscuredCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |

## LatestSiteLevelParkingAttributeParkingUsageStddevSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `obscuredCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |

## LatestSiteLevelParkingAttributeParkingUsageSumAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `BigIntFilter` |  |
| `parkingAttributeId` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `availableCount` | `BigIntFilter` |  |
| `noDataCount` | `BigIntFilter` |  |
| `obscuredCount` | `BigIntFilter` |  |
| `occupiedCount` | `BigIntFilter` |  |
| `totalCount` | `BigIntFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |

## LatestSiteLevelParkingAttributeParkingUsageVariancePopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `obscuredCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |

## LatestSiteLevelParkingAttributeParkingUsageVarianceSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `obscuredCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |

## LatestSiteLevelParkingAttributeParkingUsagesConnection (object)

Description A connection to a list of LatestSiteLevelParkingAttributeParkingUsage values.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodes` | `[LatestSiteLevelParkingAttributeParkingUsage!]!` |  |
| `totalCount` | `Int!` |  |

## LatestSiteLevelParkingAttributeParkingUsagesOrderBy (enum)

Description Methods to use when ordering LatestSiteLevelParkingAttributeParkingUsage .

## LatestSiteParkingAttributeParkingUsage (object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `Datetime` |  |
| `siteId` | `Int` |  |
| `parkingAttributeId` | `Int` |  |
| `availableCount` | `Int` |  |
| `noDataCount` | `Int` |  |
| `obscuredCount` | `Int` |  |
| `occupiedCount` | `Int` |  |
| `totalCount` | `Int` |  |
| `utilizedPercent` | `BigFloat` |  |
| `site` | `Site` |  |
| `parkingAttribute` | `ParkingAttribute` |  |

## LatestSiteParkingAttributeParkingUsageAggregatesFilter (input-object)

Description A filter to be used against aggregates of LatestSiteParkingAttributeParkingUsage object types.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `sum` | `LatestSiteParkingAttributeParkingUsageSumAggregateFilter` |  |
| `distinctCount` | `LatestSiteParkingAttributeParkingUsageDistinctCountAggregateFilter` |  |
| `min` | `LatestSiteParkingAttributeParkingUsageMinAggregateFilter` |  |
| `max` | `LatestSiteParkingAttributeParkingUsageMaxAggregateFilter` |  |
| `average` | `LatestSiteParkingAttributeParkingUsageAverageAggregateFilter` |  |
| `stddevSample` | `LatestSiteParkingAttributeParkingUsageStddevSampleAggregateFilter` |  |
| `stddevPopulation` | `LatestSiteParkingAttributeParkingUsageStddevPopulationAggregateFilter` |  |
| `varianceSample` | `LatestSiteParkingAttributeParkingUsageVarianceSampleAggregateFilter` |  |
| `variancePopulation` | `LatestSiteParkingAttributeParkingUsageVariancePopulationAggregateFilter` |  |

## LatestSiteParkingAttributeParkingUsageAverageAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `obscuredCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |

## LatestSiteParkingAttributeParkingUsageCondition (input-object)

Description A condition to be used against LatestSiteParkingAttributeParkingUsage object types. All fields are tested for equality and combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `Datetime` |  |
| `siteId` | `Int` |  |
| `parkingAttributeId` | `Int` |  |
| `availableCount` | `Int` |  |
| `noDataCount` | `Int` |  |
| `obscuredCount` | `Int` |  |
| `occupiedCount` | `Int` |  |
| `totalCount` | `Int` |  |
| `utilizedPercent` | `BigFloat` |  |

## LatestSiteParkingAttributeParkingUsageDistinctCountAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `parkingAttributeId` | `BigIntFilter` |  |
| `availableCount` | `BigIntFilter` |  |
| `noDataCount` | `BigIntFilter` |  |
| `obscuredCount` | `BigIntFilter` |  |
| `occupiedCount` | `BigIntFilter` |  |
| `totalCount` | `BigIntFilter` |  |
| `utilizedPercent` | `BigIntFilter` |  |

## LatestSiteParkingAttributeParkingUsageFilter (input-object)

Description A filter to be used against LatestSiteParkingAttributeParkingUsage object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DatetimeFilter` |  |
| `siteId` | `IntFilter` |  |
| `parkingAttributeId` | `IntFilter` |  |
| `availableCount` | `IntFilter` |  |
| `noDataCount` | `IntFilter` |  |
| `obscuredCount` | `IntFilter` |  |
| `occupiedCount` | `IntFilter` |  |
| `totalCount` | `IntFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |
| `siteExists` | `Boolean` |  |
| `parkingAttributeExists` | `Boolean` |  |

## LatestSiteParkingAttributeParkingUsageHavingAverageInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteParkingAttributeParkingUsageHavingDistinctCountInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteParkingAttributeParkingUsageHavingInput (input-object)

Description Conditions for LatestSiteParkingAttributeParkingUsage aggregates.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `AND` | `[LatestSiteParkingAttributeParkingUsageHavingInput!]` |  |
| `OR` | `[LatestSiteParkingAttributeParkingUsageHavingInput!]` |  |
| `sum` | `LatestSiteParkingAttributeParkingUsageHavingSumInput` |  |
| `distinctCount` | `LatestSiteParkingAttributeParkingUsageHavingDistinctCountInput` |  |
| `min` | `LatestSiteParkingAttributeParkingUsageHavingMinInput` |  |
| `max` | `LatestSiteParkingAttributeParkingUsageHavingMaxInput` |  |
| `average` | `LatestSiteParkingAttributeParkingUsageHavingAverageInput` |  |
| `stddevSample` | `LatestSiteParkingAttributeParkingUsageHavingStddevSampleInput` |  |
| `stddevPopulation` | `LatestSiteParkingAttributeParkingUsageHavingStddevPopulationInput` |  |
| `varianceSample` | `LatestSiteParkingAttributeParkingUsageHavingVarianceSampleInput` |  |
| `variancePopulation` | `LatestSiteParkingAttributeParkingUsageHavingVariancePopulationInput` |  |

## LatestSiteParkingAttributeParkingUsageHavingMaxInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteParkingAttributeParkingUsageHavingMinInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteParkingAttributeParkingUsageHavingStddevPopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteParkingAttributeParkingUsageHavingStddevSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteParkingAttributeParkingUsageHavingSumInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteParkingAttributeParkingUsageHavingVariancePopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteParkingAttributeParkingUsageHavingVarianceSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteParkingAttributeParkingUsageMaxAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DatetimeFilter` |  |
| `siteId` | `IntFilter` |  |
| `parkingAttributeId` | `IntFilter` |  |
| `availableCount` | `IntFilter` |  |
| `noDataCount` | `IntFilter` |  |
| `obscuredCount` | `IntFilter` |  |
| `occupiedCount` | `IntFilter` |  |
| `totalCount` | `IntFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |

## LatestSiteParkingAttributeParkingUsageMinAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DatetimeFilter` |  |
| `siteId` | `IntFilter` |  |
| `parkingAttributeId` | `IntFilter` |  |
| `availableCount` | `IntFilter` |  |
| `noDataCount` | `IntFilter` |  |
| `obscuredCount` | `IntFilter` |  |
| `occupiedCount` | `IntFilter` |  |
| `totalCount` | `IntFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |

## LatestSiteParkingAttributeParkingUsageStddevPopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `obscuredCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |

## LatestSiteParkingAttributeParkingUsageStddevSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `obscuredCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |

## LatestSiteParkingAttributeParkingUsageSumAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `BigIntFilter` |  |
| `parkingAttributeId` | `BigIntFilter` |  |
| `availableCount` | `BigIntFilter` |  |
| `noDataCount` | `BigIntFilter` |  |
| `obscuredCount` | `BigIntFilter` |  |
| `occupiedCount` | `BigIntFilter` |  |
| `totalCount` | `BigIntFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |

## LatestSiteParkingAttributeParkingUsageVariancePopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `obscuredCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |

## LatestSiteParkingAttributeParkingUsageVarianceSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `obscuredCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |

## LatestSiteParkingAttributeParkingUsagesConnection (object)

Description A connection to a list of LatestSiteParkingAttributeParkingUsage values.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodes` | `[LatestSiteParkingAttributeParkingUsage!]!` |  |
| `totalCount` | `Int!` |  |

## LatestSiteParkingAttributeParkingUsagesOrderBy (enum)

Description Methods to use when ordering LatestSiteParkingAttributeParkingUsage .

## LatestSiteParkingUsage (object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `Datetime` |  |
| `siteId` | `Int` |  |
| `availableCount` | `Int` |  |
| `noDataCount` | `Int` |  |
| `obscuredCount` | `Int` |  |
| `occupiedCount` | `Int` |  |
| `totalCount` | `Int` |  |
| `utilizedPercent` | `BigFloat` |  |
| `site` | `Site` |  |

## LatestSiteParkingUsageAggregatesFilter (input-object)

Description A filter to be used against aggregates of LatestSiteParkingUsage object types.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `sum` | `LatestSiteParkingUsageSumAggregateFilter` |  |
| `distinctCount` | `LatestSiteParkingUsageDistinctCountAggregateFilter` |  |
| `min` | `LatestSiteParkingUsageMinAggregateFilter` |  |
| `max` | `LatestSiteParkingUsageMaxAggregateFilter` |  |
| `average` | `LatestSiteParkingUsageAverageAggregateFilter` |  |
| `stddevSample` | `LatestSiteParkingUsageStddevSampleAggregateFilter` |  |
| `stddevPopulation` | `LatestSiteParkingUsageStddevPopulationAggregateFilter` |  |
| `varianceSample` | `LatestSiteParkingUsageVarianceSampleAggregateFilter` |  |
| `variancePopulation` | `LatestSiteParkingUsageVariancePopulationAggregateFilter` |  |

## LatestSiteParkingUsageAverageAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `obscuredCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |

## LatestSiteParkingUsageCondition (input-object)

Description A condition to be used against LatestSiteParkingUsage object types. All fields are tested for equality and combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `Datetime` |  |
| `siteId` | `Int` |  |
| `availableCount` | `Int` |  |
| `noDataCount` | `Int` |  |
| `obscuredCount` | `Int` |  |
| `occupiedCount` | `Int` |  |
| `totalCount` | `Int` |  |
| `utilizedPercent` | `BigFloat` |  |

## LatestSiteParkingUsageDistinctCountAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `availableCount` | `BigIntFilter` |  |
| `noDataCount` | `BigIntFilter` |  |
| `obscuredCount` | `BigIntFilter` |  |
| `occupiedCount` | `BigIntFilter` |  |
| `totalCount` | `BigIntFilter` |  |
| `utilizedPercent` | `BigIntFilter` |  |

## LatestSiteParkingUsageFilter (input-object)

Description A filter to be used against LatestSiteParkingUsage object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DatetimeFilter` |  |
| `siteId` | `IntFilter` |  |
| `availableCount` | `IntFilter` |  |
| `noDataCount` | `IntFilter` |  |
| `obscuredCount` | `IntFilter` |  |
| `occupiedCount` | `IntFilter` |  |
| `totalCount` | `IntFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |
| `siteExists` | `Boolean` |  |

## LatestSiteParkingUsageHavingAverageInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteParkingUsageHavingDistinctCountInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteParkingUsageHavingInput (input-object)

Description Conditions for LatestSiteParkingUsage aggregates.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `AND` | `[LatestSiteParkingUsageHavingInput!]` |  |
| `OR` | `[LatestSiteParkingUsageHavingInput!]` |  |
| `sum` | `LatestSiteParkingUsageHavingSumInput` |  |
| `distinctCount` | `LatestSiteParkingUsageHavingDistinctCountInput` |  |
| `min` | `LatestSiteParkingUsageHavingMinInput` |  |
| `max` | `LatestSiteParkingUsageHavingMaxInput` |  |
| `average` | `LatestSiteParkingUsageHavingAverageInput` |  |
| `stddevSample` | `LatestSiteParkingUsageHavingStddevSampleInput` |  |
| `stddevPopulation` | `LatestSiteParkingUsageHavingStddevPopulationInput` |  |
| `varianceSample` | `LatestSiteParkingUsageHavingVarianceSampleInput` |  |
| `variancePopulation` | `LatestSiteParkingUsageHavingVariancePopulationInput` |  |

## LatestSiteParkingUsageHavingMaxInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteParkingUsageHavingMinInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteParkingUsageHavingStddevPopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteParkingUsageHavingStddevSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteParkingUsageHavingSumInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteParkingUsageHavingVariancePopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteParkingUsageHavingVarianceSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `noDataCount` | `HavingIntFilter` |  |
| `obscuredCount` | `HavingIntFilter` |  |
| `occupiedCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |

## LatestSiteParkingUsageMaxAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DatetimeFilter` |  |
| `siteId` | `IntFilter` |  |
| `availableCount` | `IntFilter` |  |
| `noDataCount` | `IntFilter` |  |
| `obscuredCount` | `IntFilter` |  |
| `occupiedCount` | `IntFilter` |  |
| `totalCount` | `IntFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |

## LatestSiteParkingUsageMinAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DatetimeFilter` |  |
| `siteId` | `IntFilter` |  |
| `availableCount` | `IntFilter` |  |
| `noDataCount` | `IntFilter` |  |
| `obscuredCount` | `IntFilter` |  |
| `occupiedCount` | `IntFilter` |  |
| `totalCount` | `IntFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |

## LatestSiteParkingUsageStddevPopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `obscuredCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |

## LatestSiteParkingUsageStddevSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `obscuredCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |

## LatestSiteParkingUsageSumAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `BigIntFilter` |  |
| `availableCount` | `BigIntFilter` |  |
| `noDataCount` | `BigIntFilter` |  |
| `obscuredCount` | `BigIntFilter` |  |
| `occupiedCount` | `BigIntFilter` |  |
| `totalCount` | `BigIntFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |

## LatestSiteParkingUsageVariancePopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `obscuredCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |

## LatestSiteParkingUsageVarianceSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `obscuredCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `utilizedPercent` | `BigFloatFilter` |  |

## LatestSiteParkingUsagesConnection (object)

Description A connection to a list of LatestSiteParkingUsage values.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodes` | `[LatestSiteParkingUsage!]!` |  |
| `totalCount` | `Int!` |  |

## LatestSiteParkingUsagesOrderBy (enum)

Description Methods to use when ordering LatestSiteParkingUsage .

## Node (interface)

Description An object with a globally unique ID .

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |

## Organization (object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |
| `id` | `Int!` |  |
| `name` | `String!` |  |
| `displayName` | `String` |  |
| `description` | `String` |  |
| `createdTimestamp` | `Datetime!` |  |
| `createdUserId` | `Int` |  |
| `createdClientId` | `Int!` |  |
| `lastModifiedTimestamp` | `Datetime!` |  |
| `lastModifiedUserId` | `Int` |  |
| `lastModifiedClientId` | `Int!` |  |
| `isDeleted` | `Boolean!` |  |
| `createdUser` | `User` |  |
| `lastModifiedUser` | `User` |  |
| `organizationUsers` | `OrganizationUsersConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [OrganizationUsersOrderBy!]: [OrganizationUsersOrderBy!]`, `condition - OrganizationUserCondition: OrganizationUserCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - OrganizationUserFilter: OrganizationUserFilter` |
| `sites` | `SitesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [SitesOrderBy!]: [SitesOrderBy!]`, `condition - SiteCondition: SiteCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - SiteFilter: SiteFilter` |

## OrganizationAggregatesFilter (input-object)

Description A filter to be used against aggregates of Organization object types.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `sum` | `OrganizationSumAggregateFilter` |  |
| `distinctCount` | `OrganizationDistinctCountAggregateFilter` |  |
| `min` | `OrganizationMinAggregateFilter` |  |
| `max` | `OrganizationMaxAggregateFilter` |  |
| `average` | `OrganizationAverageAggregateFilter` |  |
| `stddevSample` | `OrganizationStddevSampleAggregateFilter` |  |
| `stddevPopulation` | `OrganizationStddevPopulationAggregateFilter` |  |
| `varianceSample` | `OrganizationVarianceSampleAggregateFilter` |  |
| `variancePopulation` | `OrganizationVariancePopulationAggregateFilter` |  |

## OrganizationAverageAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |

## OrganizationCondition (input-object)

Description A condition to be used against Organization object types. All fields are tested for equality and combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `Int` |  |
| `name` | `String` |  |
| `displayName` | `String` |  |
| `description` | `String` |  |
| `createdTimestamp` | `Datetime` |  |
| `createdUserId` | `Int` |  |
| `createdClientId` | `Int` |  |
| `lastModifiedTimestamp` | `Datetime` |  |
| `lastModifiedUserId` | `Int` |  |
| `lastModifiedClientId` | `Int` |  |
| `isDeleted` | `Boolean` |  |

## OrganizationDistinctCountAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `name` | `BigIntFilter` |  |
| `displayName` | `BigIntFilter` |  |
| `description` | `BigIntFilter` |  |
| `createdTimestamp` | `BigIntFilter` |  |
| `createdUserId` | `BigIntFilter` |  |
| `createdClientId` | `BigIntFilter` |  |
| `lastModifiedTimestamp` | `BigIntFilter` |  |
| `lastModifiedUserId` | `BigIntFilter` |  |
| `lastModifiedClientId` | `BigIntFilter` |  |
| `isDeleted` | `BigIntFilter` |  |

## OrganizationFilter (input-object)

Description A filter to be used against Organization object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `name` | `StringFilter` |  |
| `displayName` | `StringFilter` |  |
| `description` | `StringFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |
| `isDeleted` | `BooleanFilter` |  |
| `organizationUsers` | `OrganizationToManyOrganizationUserFilter` |  |
| `organizationUsersExist` | `Boolean` |  |
| `organizationPublicApiQueriesExist` | `Boolean` |  |
| `sites` | `OrganizationToManySiteFilter` |  |
| `sitesExist` | `Boolean` |  |
| `plateListsExist` | `Boolean` |  |
| `plateListEntriesExist` | `Boolean` |  |
| `platesExist` | `Boolean` |  |
| `createdUserExists` | `Boolean` |  |
| `lastModifiedUserExists` | `Boolean` |  |

## OrganizationHavingAverageInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## OrganizationHavingDistinctCountInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## OrganizationHavingInput (input-object)

Description Conditions for Organization aggregates.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `AND` | `[OrganizationHavingInput!]` |  |
| `OR` | `[OrganizationHavingInput!]` |  |
| `sum` | `OrganizationHavingSumInput` |  |
| `distinctCount` | `OrganizationHavingDistinctCountInput` |  |
| `min` | `OrganizationHavingMinInput` |  |
| `max` | `OrganizationHavingMaxInput` |  |
| `average` | `OrganizationHavingAverageInput` |  |
| `stddevSample` | `OrganizationHavingStddevSampleInput` |  |
| `stddevPopulation` | `OrganizationHavingStddevPopulationInput` |  |
| `varianceSample` | `OrganizationHavingVarianceSampleInput` |  |
| `variancePopulation` | `OrganizationHavingVariancePopulationInput` |  |

## OrganizationHavingMaxInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## OrganizationHavingMinInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## OrganizationHavingStddevPopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## OrganizationHavingStddevSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## OrganizationHavingSumInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## OrganizationHavingVariancePopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## OrganizationHavingVarianceSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## OrganizationMaxAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |

## OrganizationMinAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |

## OrganizationRole (object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |
| `id` | `Int!` |  |
| `name` | `String!` |  |
| `displayName` | `String` |  |
| `description` | `String` |  |
| `isSystem` | `Boolean!` |  |
| `createdTimestamp` | `Datetime!` |  |
| `createdUserId` | `Int` |  |
| `createdClientId` | `Int!` |  |
| `lastModifiedTimestamp` | `Datetime!` |  |
| `lastModifiedUserId` | `Int` |  |
| `lastModifiedClientId` | `Int!` |  |
| `isDeleted` | `Boolean!` |  |
| `level` | `Int!` |  |
| `createdUser` | `User` |  |
| `lastModifiedUser` | `User` |  |
| `organizationUsers` | `OrganizationUsersConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [OrganizationUsersOrderBy!]: [OrganizationUsersOrderBy!]`, `condition - OrganizationUserCondition: OrganizationUserCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - OrganizationUserFilter: OrganizationUserFilter` |

## OrganizationRoleAggregatesFilter (input-object)

Description A filter to be used against aggregates of OrganizationRole object types.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `sum` | `OrganizationRoleSumAggregateFilter` |  |
| `distinctCount` | `OrganizationRoleDistinctCountAggregateFilter` |  |
| `min` | `OrganizationRoleMinAggregateFilter` |  |
| `max` | `OrganizationRoleMaxAggregateFilter` |  |
| `average` | `OrganizationRoleAverageAggregateFilter` |  |
| `stddevSample` | `OrganizationRoleStddevSampleAggregateFilter` |  |
| `stddevPopulation` | `OrganizationRoleStddevPopulationAggregateFilter` |  |
| `varianceSample` | `OrganizationRoleVarianceSampleAggregateFilter` |  |
| `variancePopulation` | `OrganizationRoleVariancePopulationAggregateFilter` |  |

## OrganizationRoleAverageAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |
| `level` | `BigFloatFilter` |  |

## OrganizationRoleCondition (input-object)

Description A condition to be used against OrganizationRole object types. All fields are tested for equality and combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `Int` |  |
| `name` | `String` |  |
| `displayName` | `String` |  |
| `description` | `String` |  |
| `isSystem` | `Boolean` |  |
| `createdTimestamp` | `Datetime` |  |
| `createdUserId` | `Int` |  |
| `createdClientId` | `Int` |  |
| `lastModifiedTimestamp` | `Datetime` |  |
| `lastModifiedUserId` | `Int` |  |
| `lastModifiedClientId` | `Int` |  |
| `isDeleted` | `Boolean` |  |
| `level` | `Int` |  |

## OrganizationRoleDistinctCountAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `name` | `BigIntFilter` |  |
| `displayName` | `BigIntFilter` |  |
| `description` | `BigIntFilter` |  |
| `isSystem` | `BigIntFilter` |  |
| `createdTimestamp` | `BigIntFilter` |  |
| `createdUserId` | `BigIntFilter` |  |
| `createdClientId` | `BigIntFilter` |  |
| `lastModifiedTimestamp` | `BigIntFilter` |  |
| `lastModifiedUserId` | `BigIntFilter` |  |
| `lastModifiedClientId` | `BigIntFilter` |  |
| `isDeleted` | `BigIntFilter` |  |
| `level` | `BigIntFilter` |  |

## OrganizationRoleFilter (input-object)

Description A filter to be used against OrganizationRole object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `name` | `StringFilter` |  |
| `displayName` | `StringFilter` |  |
| `description` | `StringFilter` |  |
| `isSystem` | `BooleanFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |
| `isDeleted` | `BooleanFilter` |  |
| `level` | `IntFilter` |  |
| `organizationUsers` | `OrganizationRoleToManyOrganizationUserFilter` |  |
| `organizationUsersExist` | `Boolean` |  |
| `createdUserExists` | `Boolean` |  |
| `lastModifiedUserExists` | `Boolean` |  |

## OrganizationRoleHavingAverageInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `level` | `HavingIntFilter` |  |

## OrganizationRoleHavingDistinctCountInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `level` | `HavingIntFilter` |  |

## OrganizationRoleHavingInput (input-object)

Description Conditions for OrganizationRole aggregates.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `AND` | `[OrganizationRoleHavingInput!]` |  |
| `OR` | `[OrganizationRoleHavingInput!]` |  |
| `sum` | `OrganizationRoleHavingSumInput` |  |
| `distinctCount` | `OrganizationRoleHavingDistinctCountInput` |  |
| `min` | `OrganizationRoleHavingMinInput` |  |
| `max` | `OrganizationRoleHavingMaxInput` |  |
| `average` | `OrganizationRoleHavingAverageInput` |  |
| `stddevSample` | `OrganizationRoleHavingStddevSampleInput` |  |
| `stddevPopulation` | `OrganizationRoleHavingStddevPopulationInput` |  |
| `varianceSample` | `OrganizationRoleHavingVarianceSampleInput` |  |
| `variancePopulation` | `OrganizationRoleHavingVariancePopulationInput` |  |

## OrganizationRoleHavingMaxInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `level` | `HavingIntFilter` |  |

## OrganizationRoleHavingMinInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `level` | `HavingIntFilter` |  |

## OrganizationRoleHavingStddevPopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `level` | `HavingIntFilter` |  |

## OrganizationRoleHavingStddevSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `level` | `HavingIntFilter` |  |

## OrganizationRoleHavingSumInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `level` | `HavingIntFilter` |  |

## OrganizationRoleHavingVariancePopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `level` | `HavingIntFilter` |  |

## OrganizationRoleHavingVarianceSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `level` | `HavingIntFilter` |  |

## OrganizationRoleMaxAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |
| `level` | `IntFilter` |  |

## OrganizationRoleMinAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |
| `level` | `IntFilter` |  |

## OrganizationRoleStddevPopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |
| `level` | `BigFloatFilter` |  |

## OrganizationRoleStddevSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |
| `level` | `BigFloatFilter` |  |

## OrganizationRoleSumAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `createdUserId` | `BigIntFilter` |  |
| `createdClientId` | `BigIntFilter` |  |
| `lastModifiedUserId` | `BigIntFilter` |  |
| `lastModifiedClientId` | `BigIntFilter` |  |
| `level` | `BigIntFilter` |  |

## OrganizationRoleToManyOrganizationUserFilter (input-object)

Description A filter to be used against many OrganizationUser object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `OrganizationUserAggregatesFilter` |  |

## OrganizationRoleVariancePopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |
| `level` | `BigFloatFilter` |  |

## OrganizationRoleVarianceSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |
| `level` | `BigFloatFilter` |  |

## OrganizationRolesConnection (object)

Description A connection to a list of OrganizationRole values.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodes` | `[OrganizationRole!]!` |  |
| `totalCount` | `Int!` |  |

## OrganizationRolesOrderBy (enum)

Description Methods to use when ordering OrganizationRole .

## OrganizationStddevPopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |

## OrganizationStddevSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |

## OrganizationSumAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `createdUserId` | `BigIntFilter` |  |
| `createdClientId` | `BigIntFilter` |  |
| `lastModifiedUserId` | `BigIntFilter` |  |
| `lastModifiedClientId` | `BigIntFilter` |  |

## OrganizationToManyOrganizationUserFilter (input-object)

Description A filter to be used against many OrganizationUser object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `OrganizationUserAggregatesFilter` |  |

## OrganizationToManySiteFilter (input-object)

Description A filter to be used against many Site object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `SiteAggregatesFilter` |  |

## OrganizationUser (object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |
| `id` | `Int!` |  |
| `organizationId` | `Int!` |  |
| `userId` | `Int!` |  |
| `organizationRoleId` | `Int!` |  |
| `createdTimestamp` | `Datetime!` |  |
| `createdUserId` | `Int` |  |
| `createdClientId` | `Int!` |  |
| `lastModifiedTimestamp` | `Datetime!` |  |
| `lastModifiedUserId` | `Int` |  |
| `lastModifiedClientId` | `Int!` |  |
| `isDeleted` | `Boolean!` |  |
| `isEnabled` | `Boolean!` |  |
| `organization` | `Organization` |  |
| `user` | `User` |  |
| `organizationRole` | `OrganizationRole` |  |
| `createdUser` | `User` |  |
| `lastModifiedUser` | `User` |  |

## OrganizationUserAggregatesFilter (input-object)

Description A filter to be used against aggregates of OrganizationUser object types.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `sum` | `OrganizationUserSumAggregateFilter` |  |
| `distinctCount` | `OrganizationUserDistinctCountAggregateFilter` |  |
| `min` | `OrganizationUserMinAggregateFilter` |  |
| `max` | `OrganizationUserMaxAggregateFilter` |  |
| `average` | `OrganizationUserAverageAggregateFilter` |  |
| `stddevSample` | `OrganizationUserStddevSampleAggregateFilter` |  |
| `stddevPopulation` | `OrganizationUserStddevPopulationAggregateFilter` |  |
| `varianceSample` | `OrganizationUserVarianceSampleAggregateFilter` |  |
| `variancePopulation` | `OrganizationUserVariancePopulationAggregateFilter` |  |

## OrganizationUserAverageAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `organizationId` | `BigFloatFilter` |  |
| `userId` | `BigFloatFilter` |  |
| `organizationRoleId` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |

## OrganizationUserCondition (input-object)

Description A condition to be used against OrganizationUser object types. All fields are tested for equality and combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `Int` |  |
| `organizationId` | `Int` |  |
| `userId` | `Int` |  |
| `organizationRoleId` | `Int` |  |
| `createdTimestamp` | `Datetime` |  |
| `createdUserId` | `Int` |  |
| `createdClientId` | `Int` |  |
| `lastModifiedTimestamp` | `Datetime` |  |
| `lastModifiedUserId` | `Int` |  |
| `lastModifiedClientId` | `Int` |  |
| `isDeleted` | `Boolean` |  |
| `isEnabled` | `Boolean` |  |

## OrganizationUserDistinctCountAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `organizationId` | `BigIntFilter` |  |
| `userId` | `BigIntFilter` |  |
| `organizationRoleId` | `BigIntFilter` |  |
| `createdTimestamp` | `BigIntFilter` |  |
| `createdUserId` | `BigIntFilter` |  |
| `createdClientId` | `BigIntFilter` |  |
| `lastModifiedTimestamp` | `BigIntFilter` |  |
| `lastModifiedUserId` | `BigIntFilter` |  |
| `lastModifiedClientId` | `BigIntFilter` |  |
| `isDeleted` | `BigIntFilter` |  |
| `isEnabled` | `BigIntFilter` |  |

## OrganizationUserFilter (input-object)

Description A filter to be used against OrganizationUser object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `organizationId` | `IntFilter` |  |
| `userId` | `IntFilter` |  |
| `organizationRoleId` | `IntFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |
| `isDeleted` | `BooleanFilter` |  |
| `isEnabled` | `BooleanFilter` |  |
| `createdUserExists` | `Boolean` |  |
| `lastModifiedUserExists` | `Boolean` |  |

## OrganizationUserHavingAverageInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `organizationId` | `HavingIntFilter` |  |
| `userId` | `HavingIntFilter` |  |
| `organizationRoleId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## OrganizationUserHavingDistinctCountInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `organizationId` | `HavingIntFilter` |  |
| `userId` | `HavingIntFilter` |  |
| `organizationRoleId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## OrganizationUserHavingInput (input-object)

Description Conditions for OrganizationUser aggregates.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `AND` | `[OrganizationUserHavingInput!]` |  |
| `OR` | `[OrganizationUserHavingInput!]` |  |
| `sum` | `OrganizationUserHavingSumInput` |  |
| `distinctCount` | `OrganizationUserHavingDistinctCountInput` |  |
| `min` | `OrganizationUserHavingMinInput` |  |
| `max` | `OrganizationUserHavingMaxInput` |  |
| `average` | `OrganizationUserHavingAverageInput` |  |
| `stddevSample` | `OrganizationUserHavingStddevSampleInput` |  |
| `stddevPopulation` | `OrganizationUserHavingStddevPopulationInput` |  |
| `varianceSample` | `OrganizationUserHavingVarianceSampleInput` |  |
| `variancePopulation` | `OrganizationUserHavingVariancePopulationInput` |  |

## OrganizationUserHavingMaxInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `organizationId` | `HavingIntFilter` |  |
| `userId` | `HavingIntFilter` |  |
| `organizationRoleId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## OrganizationUserHavingMinInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `organizationId` | `HavingIntFilter` |  |
| `userId` | `HavingIntFilter` |  |
| `organizationRoleId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## OrganizationUserHavingStddevPopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `organizationId` | `HavingIntFilter` |  |
| `userId` | `HavingIntFilter` |  |
| `organizationRoleId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## OrganizationUserHavingStddevSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `organizationId` | `HavingIntFilter` |  |
| `userId` | `HavingIntFilter` |  |
| `organizationRoleId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## OrganizationUserHavingSumInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `organizationId` | `HavingIntFilter` |  |
| `userId` | `HavingIntFilter` |  |
| `organizationRoleId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## OrganizationUserHavingVariancePopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `organizationId` | `HavingIntFilter` |  |
| `userId` | `HavingIntFilter` |  |
| `organizationRoleId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## OrganizationUserHavingVarianceSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `organizationId` | `HavingIntFilter` |  |
| `userId` | `HavingIntFilter` |  |
| `organizationRoleId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## OrganizationUserMaxAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `organizationId` | `IntFilter` |  |
| `userId` | `IntFilter` |  |
| `organizationRoleId` | `IntFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |

## OrganizationUserMinAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `organizationId` | `IntFilter` |  |
| `userId` | `IntFilter` |  |
| `organizationRoleId` | `IntFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |

## OrganizationUserStddevPopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `organizationId` | `BigFloatFilter` |  |
| `userId` | `BigFloatFilter` |  |
| `organizationRoleId` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |

## OrganizationUserStddevSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `organizationId` | `BigFloatFilter` |  |
| `userId` | `BigFloatFilter` |  |
| `organizationRoleId` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |

## OrganizationUserSumAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `organizationId` | `BigIntFilter` |  |
| `userId` | `BigIntFilter` |  |
| `organizationRoleId` | `BigIntFilter` |  |
| `createdUserId` | `BigIntFilter` |  |
| `createdClientId` | `BigIntFilter` |  |
| `lastModifiedUserId` | `BigIntFilter` |  |
| `lastModifiedClientId` | `BigIntFilter` |  |

## OrganizationUserVariancePopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `organizationId` | `BigFloatFilter` |  |
| `userId` | `BigFloatFilter` |  |
| `organizationRoleId` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |

## OrganizationUserVarianceSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `organizationId` | `BigFloatFilter` |  |
| `userId` | `BigFloatFilter` |  |
| `organizationRoleId` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |

## OrganizationUsersConnection (object)

Description A connection to a list of OrganizationUser values.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodes` | `[OrganizationUser!]!` |  |
| `totalCount` | `Int!` |  |

## OrganizationUsersOrderBy (enum)

Description Methods to use when ordering OrganizationUser .

## OrganizationVariancePopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |

## OrganizationVarianceSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |

## OrganizationsConnection (object)

Description A connection to a list of Organization values.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodes` | `[Organization!]!` |  |
| `totalCount` | `Int!` |  |

## OrganizationsOrderBy (enum)

Description Methods to use when ordering Organization .

## ParkingAttribute (object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |
| `id` | `Int!` |  |
| `siteId` | `Int!` |  |
| `name` | `String!` |  |
| `displayName` | `String` |  |
| `isSystem` | `Boolean!` |  |
| `ledRgbaValue` | `Int!` |  |
| `displayRgbaValue` | `Int!` |  |
| `reportRgbaValue` | `Int` |  |
| `isEnabled` | `Boolean!` |  |
| `createdTimestamp` | `Datetime!` |  |
| `createdUserId` | `Int` |  |
| `createdClientId` | `Int!` |  |
| `lastModifiedTimestamp` | `Datetime!` |  |
| `lastModifiedUserId` | `Int` |  |
| `lastModifiedClientId` | `Int!` |  |
| `isDeleted` | `Boolean!` |  |
| `priority` | `Int` |  |
| `remoteId` | `Int` |  |
| `site` | `Site` |  |
| `createdUser` | `User` |  |
| `lastModifiedUser` | `User` |  |
| `parkingZoneDataPoints` | `ParkingZoneDataPointsConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingZoneDataPointsOrderBy!]: [ParkingZoneDataPointsOrderBy!]`, `condition - ParkingZoneDataPointCondition: ParkingZoneDataPointCondition`, `filter - ParkingZoneDataPointFilter: ParkingZoneDataPointFilter` |
| `siteLevelParkingAttributeUsageByHours` | `SiteLevelParkingAttributeUsageByHoursConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [SiteLevelParkingAttributeUsageByHoursOrderBy!]: [SiteLevelParkingAttributeUsageByHoursOrderBy!]`, `condition - SiteLevelParkingAttributeUsageByHourCondition: SiteLevelParkingAttributeUsageByHourCondition`, `filter - SiteLevelParkingAttributeUsageByHourFilter: SiteLevelParkingAttributeUsageByHourFilter` |
| `parkingSpaceDataPoints` | `ParkingSpaceDataPointsConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingSpaceDataPointsOrderBy!]: [ParkingSpaceDataPointsOrderBy!]`, `condition - ParkingSpaceDataPointCondition: ParkingSpaceDataPointCondition`, `filter - ParkingSpaceDataPointFilter: ParkingSpaceDataPointFilter` |
| `latestSiteLevelParkingAttributeParkingUsages` | `LatestSiteLevelParkingAttributeParkingUsagesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [LatestSiteLevelParkingAttributeParkingUsagesOrderBy!]: [LatestSiteLevelParkingAttributeParkingUsagesOrderBy!]`, `condition - LatestSiteLevelParkingAttributeParkingUsageCondition: LatestSiteLevelParkingAttributeParkingUsageCondition`, `filter - LatestSiteLevelParkingAttributeParkingUsageFilter: LatestSiteLevelParkingAttributeParkingUsageFilter` |
| `latestSiteParkingAttributeParkingUsages` | `LatestSiteParkingAttributeParkingUsagesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [LatestSiteParkingAttributeParkingUsagesOrderBy!]: [LatestSiteParkingAttributeParkingUsagesOrderBy!]`, `condition - LatestSiteParkingAttributeParkingUsageCondition: LatestSiteParkingAttributeParkingUsageCondition`, `filter - LatestSiteParkingAttributeParkingUsageFilter: LatestSiteParkingAttributeParkingUsageFilter` |
| `reportingSiteLevelParkingAttributeUsageByHours` | `ReportingSiteLevelParkingAttributeUsageByHoursConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ReportingSiteLevelParkingAttributeUsageByHoursOrderBy!]: [ReportingSiteLevelParkingAttributeUsageByHoursOrderBy!]`, `condition - ReportingSiteLevelParkingAttributeUsageByHourCondition: ReportingSiteLevelParkingAttributeUsageByHourCondition`, `filter - ReportingSiteLevelParkingAttributeUsageByHourFilter: ReportingSiteLevelParkingAttributeUsageByHourFilter` |

## ParkingAttributeAggregatesFilter (input-object)

Description A filter to be used against aggregates of ParkingAttribute object types.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `sum` | `ParkingAttributeSumAggregateFilter` |  |
| `distinctCount` | `ParkingAttributeDistinctCountAggregateFilter` |  |
| `min` | `ParkingAttributeMinAggregateFilter` |  |
| `max` | `ParkingAttributeMaxAggregateFilter` |  |
| `average` | `ParkingAttributeAverageAggregateFilter` |  |
| `stddevSample` | `ParkingAttributeStddevSampleAggregateFilter` |  |
| `stddevPopulation` | `ParkingAttributeStddevPopulationAggregateFilter` |  |
| `varianceSample` | `ParkingAttributeVarianceSampleAggregateFilter` |  |
| `variancePopulation` | `ParkingAttributeVariancePopulationAggregateFilter` |  |

## ParkingAttributeAverageAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `ledRgbaValue` | `BigFloatFilter` |  |
| `displayRgbaValue` | `BigFloatFilter` |  |
| `reportRgbaValue` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |
| `priority` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |

## ParkingAttributeCondition (input-object)

Description A condition to be used against ParkingAttribute object types. All fields are tested for equality and combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `Int` |  |
| `siteId` | `Int` |  |
| `name` | `String` |  |
| `displayName` | `String` |  |
| `isSystem` | `Boolean` |  |
| `ledRgbaValue` | `Int` |  |
| `displayRgbaValue` | `Int` |  |
| `reportRgbaValue` | `Int` |  |
| `isEnabled` | `Boolean` |  |
| `createdTimestamp` | `Datetime` |  |
| `createdUserId` | `Int` |  |
| `createdClientId` | `Int` |  |
| `lastModifiedTimestamp` | `Datetime` |  |
| `lastModifiedUserId` | `Int` |  |
| `lastModifiedClientId` | `Int` |  |
| `isDeleted` | `Boolean` |  |
| `priority` | `Int` |  |
| `remoteId` | `Int` |  |

## ParkingAttributeDistinctCountAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `name` | `BigIntFilter` |  |
| `displayName` | `BigIntFilter` |  |
| `isSystem` | `BigIntFilter` |  |
| `ledRgbaValue` | `BigIntFilter` |  |
| `displayRgbaValue` | `BigIntFilter` |  |
| `reportRgbaValue` | `BigIntFilter` |  |
| `isEnabled` | `BigIntFilter` |  |
| `createdTimestamp` | `BigIntFilter` |  |
| `createdUserId` | `BigIntFilter` |  |
| `createdClientId` | `BigIntFilter` |  |
| `lastModifiedTimestamp` | `BigIntFilter` |  |
| `lastModifiedUserId` | `BigIntFilter` |  |
| `lastModifiedClientId` | `BigIntFilter` |  |
| `isDeleted` | `BigIntFilter` |  |
| `priority` | `BigIntFilter` |  |
| `remoteId` | `BigIntFilter` |  |

## ParkingAttributeFilter (input-object)

Description A filter to be used against ParkingAttribute object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `siteId` | `IntFilter` |  |
| `name` | `StringFilter` |  |
| `displayName` | `StringFilter` |  |
| `isSystem` | `BooleanFilter` |  |
| `ledRgbaValue` | `IntFilter` |  |
| `displayRgbaValue` | `IntFilter` |  |
| `reportRgbaValue` | `IntFilter` |  |
| `isEnabled` | `BooleanFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |
| `isDeleted` | `BooleanFilter` |  |
| `priority` | `IntFilter` |  |
| `remoteId` | `IntFilter` |  |
| `parkingSpaceUtilizationEventsExist` | `Boolean` |  |
| `parkingZoneDataPoints` | `ParkingAttributeToManyParkingZoneDataPointFilter` |  |
| `parkingZoneDataPointsExist` | `Boolean` |  |
| `parkingZoneUtilizationEventsExist` | `Boolean` |  |
| `siteLevelParkingAttributeUsageByHours` | `ParkingAttributeToManySiteLevelParkingAttributeUsageByHourFilter` |  |
| `siteLevelParkingAttributeUsageByHoursExist` | `Boolean` |  |
| `snapshotParkingSpacesExist` | `Boolean` |  |
| `parkingSpaceDataPoints` | `ParkingAttributeToManyParkingSpaceDataPointFilter` |  |
| `parkingSpaceDataPointsExist` | `Boolean` |  |
| `latestSiteLevelParkingAttributeParkingUsages` | `ParkingAttributeToManyLatestSiteLevelParkingAttributeParkingUsageFilter` |  |
| `latestSiteLevelParkingAttributeParkingUsagesExist` | `Boolean` |  |
| `latestSiteParkingAttributeParkingUsages` | `ParkingAttributeToManyLatestSiteParkingAttributeParkingUsageFilter` |  |
| `latestSiteParkingAttributeParkingUsagesExist` | `Boolean` |  |
| `reportingSiteLevelParkingAttributeUsageByHours` | `ParkingAttributeToManyReportingSiteLevelParkingAttributeUsageByHourFilter` |  |
| `reportingSiteLevelParkingAttributeUsageByHoursExist` | `Boolean` |  |
| `createdUserExists` | `Boolean` |  |
| `lastModifiedUserExists` | `Boolean` |  |

## ParkingAttributeHavingAverageInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `ledRgbaValue` | `HavingIntFilter` |  |
| `displayRgbaValue` | `HavingIntFilter` |  |
| `reportRgbaValue` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `priority` | `HavingIntFilter` |  |
| `remoteId` | `HavingIntFilter` |  |

## ParkingAttributeHavingDistinctCountInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `ledRgbaValue` | `HavingIntFilter` |  |
| `displayRgbaValue` | `HavingIntFilter` |  |
| `reportRgbaValue` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `priority` | `HavingIntFilter` |  |
| `remoteId` | `HavingIntFilter` |  |

## ParkingAttributeHavingInput (input-object)

Description Conditions for ParkingAttribute aggregates.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `AND` | `[ParkingAttributeHavingInput!]` |  |
| `OR` | `[ParkingAttributeHavingInput!]` |  |
| `sum` | `ParkingAttributeHavingSumInput` |  |
| `distinctCount` | `ParkingAttributeHavingDistinctCountInput` |  |
| `min` | `ParkingAttributeHavingMinInput` |  |
| `max` | `ParkingAttributeHavingMaxInput` |  |
| `average` | `ParkingAttributeHavingAverageInput` |  |
| `stddevSample` | `ParkingAttributeHavingStddevSampleInput` |  |
| `stddevPopulation` | `ParkingAttributeHavingStddevPopulationInput` |  |
| `varianceSample` | `ParkingAttributeHavingVarianceSampleInput` |  |
| `variancePopulation` | `ParkingAttributeHavingVariancePopulationInput` |  |

## ParkingAttributeHavingMaxInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `ledRgbaValue` | `HavingIntFilter` |  |
| `displayRgbaValue` | `HavingIntFilter` |  |
| `reportRgbaValue` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `priority` | `HavingIntFilter` |  |
| `remoteId` | `HavingIntFilter` |  |

## ParkingAttributeHavingMinInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `ledRgbaValue` | `HavingIntFilter` |  |
| `displayRgbaValue` | `HavingIntFilter` |  |
| `reportRgbaValue` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `priority` | `HavingIntFilter` |  |
| `remoteId` | `HavingIntFilter` |  |

## ParkingAttributeHavingStddevPopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `ledRgbaValue` | `HavingIntFilter` |  |
| `displayRgbaValue` | `HavingIntFilter` |  |
| `reportRgbaValue` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `priority` | `HavingIntFilter` |  |
| `remoteId` | `HavingIntFilter` |  |

## ParkingAttributeHavingStddevSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `ledRgbaValue` | `HavingIntFilter` |  |
| `displayRgbaValue` | `HavingIntFilter` |  |
| `reportRgbaValue` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `priority` | `HavingIntFilter` |  |
| `remoteId` | `HavingIntFilter` |  |

## ParkingAttributeHavingSumInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `ledRgbaValue` | `HavingIntFilter` |  |
| `displayRgbaValue` | `HavingIntFilter` |  |
| `reportRgbaValue` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `priority` | `HavingIntFilter` |  |
| `remoteId` | `HavingIntFilter` |  |

## ParkingAttributeHavingVariancePopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `ledRgbaValue` | `HavingIntFilter` |  |
| `displayRgbaValue` | `HavingIntFilter` |  |
| `reportRgbaValue` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `priority` | `HavingIntFilter` |  |
| `remoteId` | `HavingIntFilter` |  |

## ParkingAttributeHavingVarianceSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `ledRgbaValue` | `HavingIntFilter` |  |
| `displayRgbaValue` | `HavingIntFilter` |  |
| `reportRgbaValue` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `priority` | `HavingIntFilter` |  |
| `remoteId` | `HavingIntFilter` |  |

## ParkingAttributeMaxAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `siteId` | `IntFilter` |  |
| `ledRgbaValue` | `IntFilter` |  |
| `displayRgbaValue` | `IntFilter` |  |
| `reportRgbaValue` | `IntFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |
| `priority` | `IntFilter` |  |
| `remoteId` | `IntFilter` |  |

## ParkingAttributeMinAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `siteId` | `IntFilter` |  |
| `ledRgbaValue` | `IntFilter` |  |
| `displayRgbaValue` | `IntFilter` |  |
| `reportRgbaValue` | `IntFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |
| `priority` | `IntFilter` |  |
| `remoteId` | `IntFilter` |  |

## ParkingAttributeStddevPopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `ledRgbaValue` | `BigFloatFilter` |  |
| `displayRgbaValue` | `BigFloatFilter` |  |
| `reportRgbaValue` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |
| `priority` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |

## ParkingAttributeStddevSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `ledRgbaValue` | `BigFloatFilter` |  |
| `displayRgbaValue` | `BigFloatFilter` |  |
| `reportRgbaValue` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |
| `priority` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |

## ParkingAttributeSumAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `ledRgbaValue` | `BigIntFilter` |  |
| `displayRgbaValue` | `BigIntFilter` |  |
| `reportRgbaValue` | `BigIntFilter` |  |
| `createdUserId` | `BigIntFilter` |  |
| `createdClientId` | `BigIntFilter` |  |
| `lastModifiedUserId` | `BigIntFilter` |  |
| `lastModifiedClientId` | `BigIntFilter` |  |
| `priority` | `BigIntFilter` |  |
| `remoteId` | `BigIntFilter` |  |

## ParkingAttributeToManyLatestSiteLevelParkingAttributeParkingUsageFilter (input-object)

Description A filter to be used against many LatestSiteLevelParkingAttributeParkingUsage object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `LatestSiteLevelParkingAttributeParkingUsageAggregatesFilter` |  |

## ParkingAttributeToManyLatestSiteParkingAttributeParkingUsageFilter (input-object)

Description A filter to be used against many LatestSiteParkingAttributeParkingUsage object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `LatestSiteParkingAttributeParkingUsageAggregatesFilter` |  |

## ParkingAttributeToManyParkingSpaceDataPointFilter (input-object)

Description A filter to be used against many ParkingSpaceDataPoint object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingSpaceDataPointAggregatesFilter` |  |

## ParkingAttributeToManyParkingZoneDataPointFilter (input-object)

Description A filter to be used against many ParkingZoneDataPoint object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingZoneDataPointAggregatesFilter` |  |

## ParkingAttributeToManyReportingSiteLevelParkingAttributeUsageByHourFilter (input-object)

Description A filter to be used against many ReportingSiteLevelParkingAttributeUsageByHour object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ReportingSiteLevelParkingAttributeUsageByHourAggregatesFilter` |  |

## ParkingAttributeToManySiteLevelParkingAttributeUsageByHourFilter (input-object)

Description A filter to be used against many SiteLevelParkingAttributeUsageByHour object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `SiteLevelParkingAttributeUsageByHourAggregatesFilter` |  |

## ParkingAttributeVariancePopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `ledRgbaValue` | `BigFloatFilter` |  |
| `displayRgbaValue` | `BigFloatFilter` |  |
| `reportRgbaValue` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |
| `priority` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |

## ParkingAttributeVarianceSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `ledRgbaValue` | `BigFloatFilter` |  |
| `displayRgbaValue` | `BigFloatFilter` |  |
| `reportRgbaValue` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |
| `priority` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |

## ParkingAttributesConnection (object)

Description A connection to a list of ParkingAttribute values.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodes` | `[ParkingAttribute!]!` |  |
| `totalCount` | `Int!` |  |

## ParkingAttributesOrderBy (enum)

Description Methods to use when ordering ParkingAttribute .

## ParkingSpace (object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |
| `id` | `Int!` |  |
| `siteId` | `Int!` |  |
| `siteLevelId` | `Int!` |  |
| `name` | `String!` |  |
| `displayName` | `String` |  |
| `description` | `String` |  |
| `createdTimestamp` | `Datetime!` |  |
| `createdUserId` | `Int` |  |
| `createdClientId` | `Int!` |  |
| `lastModifiedTimestamp` | `Datetime!` |  |
| `lastModifiedUserId` | `Int` |  |
| `lastModifiedClientId` | `Int!` |  |
| `isDeleted` | `Boolean!` |  |
| `indicatedBySensorIdOld` | `String` |  |
| `detectedBySensorIdOld` | `String` |  |
| `isEnabled` | `Boolean!` |  |
| `indicatedBySensorId` | `BigInt` |  |
| `detectedBySensorId` | `BigInt` |  |
| `site` | `Site` |  |
| `siteLevel` | `SiteLevel` |  |
| `createdUser` | `User` |  |
| `lastModifiedUser` | `User` |  |
| `indicatedBySensor` | `Sensor` |  |
| `detectedBySensor` | `Sensor` |  |
| `parkingSpaceDataPoints` | `ParkingSpaceDataPointsConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingSpaceDataPointsOrderBy!]: [ParkingSpaceDataPointsOrderBy!]`, `condition - ParkingSpaceDataPointCondition: ParkingSpaceDataPointCondition`, `filter - ParkingSpaceDataPointFilter: ParkingSpaceDataPointFilter` |
| `parkingSpaceVehicleSessions` | `ParkingSpaceVehicleSessionsConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingSpaceVehicleSessionsOrderBy!]: [ParkingSpaceVehicleSessionsOrderBy!]`, `condition - ParkingSpaceVehicleSessionCondition: ParkingSpaceVehicleSessionCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - ParkingSpaceVehicleSessionFilter: ParkingSpaceVehicleSessionFilter` |

## ParkingSpaceAggregatesFilter (input-object)

Description A filter to be used against aggregates of ParkingSpace object types.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `sum` | `ParkingSpaceSumAggregateFilter` |  |
| `distinctCount` | `ParkingSpaceDistinctCountAggregateFilter` |  |
| `min` | `ParkingSpaceMinAggregateFilter` |  |
| `max` | `ParkingSpaceMaxAggregateFilter` |  |
| `average` | `ParkingSpaceAverageAggregateFilter` |  |
| `stddevSample` | `ParkingSpaceStddevSampleAggregateFilter` |  |
| `stddevPopulation` | `ParkingSpaceStddevPopulationAggregateFilter` |  |
| `varianceSample` | `ParkingSpaceVarianceSampleAggregateFilter` |  |
| `variancePopulation` | `ParkingSpaceVariancePopulationAggregateFilter` |  |

## ParkingSpaceAverageAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `siteLevelId` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |
| `indicatedBySensorId` | `BigFloatFilter` |  |
| `detectedBySensorId` | `BigFloatFilter` |  |

## ParkingSpaceCondition (input-object)

Description A condition to be used against ParkingSpace object types. All fields are tested for equality and combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `Int` |  |
| `siteId` | `Int` |  |
| `siteLevelId` | `Int` |  |
| `name` | `String` |  |
| `displayName` | `String` |  |
| `description` | `String` |  |
| `createdTimestamp` | `Datetime` |  |
| `createdUserId` | `Int` |  |
| `createdClientId` | `Int` |  |
| `lastModifiedTimestamp` | `Datetime` |  |
| `lastModifiedUserId` | `Int` |  |
| `lastModifiedClientId` | `Int` |  |
| `isDeleted` | `Boolean` |  |
| `indicatedBySensorIdOld` | `String` |  |
| `detectedBySensorIdOld` | `String` |  |
| `isEnabled` | `Boolean` |  |
| `indicatedBySensorId` | `BigInt` |  |
| `detectedBySensorId` | `BigInt` |  |

## ParkingSpaceDataPoint (object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |
| `id` | `BigInt!` |  |
| `parkingSpaceId` | `Int!` |  |
| `siteLevelId` | `Int!` |  |
| `parkingAttributeId` | `Int!` |  |
| `siteId` | `Int!` |  |
| `occupancyStatus` | `Int!` |  |
| `period` | `DatetimeRange!` |  |
| `duration` | `Int` |  |
| `parkingSpaceUtilizationEventRemoteId` | `BigInt` |  |
| `occupancyPeriod` | `DatetimeRange` |  |
| `occupancyDuration` | `Int` |  |
| `occupancyStartParkingSpaceDataPointId` | `BigInt` |  |
| `parkingSpace` | `ParkingSpace` |  |
| `siteLevel` | `SiteLevel` |  |
| `parkingAttribute` | `ParkingAttribute` |  |
| `site` | `Site` |  |
| `siteOccupancyStartParkingSpaceDataPoint` | `ParkingSpaceDataPoint` |  |
| `parkingSpaceDataPointsBySiteIdAndOccupancyStartParkingSpaceDataPointId` | `ParkingSpaceDataPointsConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingSpaceDataPointsOrderBy!]: [ParkingSpaceDataPointsOrderBy!]`, `condition - ParkingSpaceDataPointCondition: ParkingSpaceDataPointCondition`, `filter - ParkingSpaceDataPointFilter: ParkingSpaceDataPointFilter` |

## ParkingSpaceDataPointAggregatesFilter (input-object)

Description A filter to be used against aggregates of ParkingSpaceDataPoint object types.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `sum` | `ParkingSpaceDataPointSumAggregateFilter` |  |
| `distinctCount` | `ParkingSpaceDataPointDistinctCountAggregateFilter` |  |
| `min` | `ParkingSpaceDataPointMinAggregateFilter` |  |
| `max` | `ParkingSpaceDataPointMaxAggregateFilter` |  |
| `average` | `ParkingSpaceDataPointAverageAggregateFilter` |  |
| `stddevSample` | `ParkingSpaceDataPointStddevSampleAggregateFilter` |  |
| `stddevPopulation` | `ParkingSpaceDataPointStddevPopulationAggregateFilter` |  |
| `varianceSample` | `ParkingSpaceDataPointVarianceSampleAggregateFilter` |  |
| `variancePopulation` | `ParkingSpaceDataPointVariancePopulationAggregateFilter` |  |

## ParkingSpaceDataPointAverageAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `parkingSpaceId` | `BigFloatFilter` |  |
| `siteLevelId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `occupancyStatus` | `BigFloatFilter` |  |
| `duration` | `BigFloatFilter` |  |
| `parkingSpaceUtilizationEventRemoteId` | `BigFloatFilter` |  |
| `occupancyDuration` | `BigFloatFilter` |  |
| `occupancyStartParkingSpaceDataPointId` | `BigFloatFilter` |  |

## ParkingSpaceDataPointCondition (input-object)

Description A condition to be used against ParkingSpaceDataPoint object types. All fields are tested for equality and combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigInt` |  |
| `parkingSpaceId` | `Int` |  |
| `siteLevelId` | `Int` |  |
| `parkingAttributeId` | `Int` |  |
| `siteId` | `Int` |  |
| `occupancyStatus` | `Int` |  |
| `period` | `DatetimeRangeInput` |  |
| `duration` | `Int` |  |
| `parkingSpaceUtilizationEventRemoteId` | `BigInt` |  |
| `occupancyPeriod` | `DatetimeRangeInput` |  |
| `occupancyDuration` | `Int` |  |
| `occupancyStartParkingSpaceDataPointId` | `BigInt` |  |

## ParkingSpaceDataPointDistinctCountAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `parkingSpaceId` | `BigIntFilter` |  |
| `siteLevelId` | `BigIntFilter` |  |
| `parkingAttributeId` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `occupancyStatus` | `BigIntFilter` |  |
| `period` | `BigIntFilter` |  |
| `duration` | `BigIntFilter` |  |
| `parkingSpaceUtilizationEventRemoteId` | `BigIntFilter` |  |
| `occupancyPeriod` | `BigIntFilter` |  |
| `occupancyDuration` | `BigIntFilter` |  |
| `occupancyStartParkingSpaceDataPointId` | `BigIntFilter` |  |

## ParkingSpaceDataPointFilter (input-object)

Description A filter to be used against ParkingSpaceDataPoint object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `parkingSpaceId` | `IntFilter` |  |
| `siteLevelId` | `IntFilter` |  |
| `parkingAttributeId` | `IntFilter` |  |
| `siteId` | `IntFilter` |  |
| `occupancyStatus` | `IntFilter` |  |
| `period` | `DatetimeRangeFilter` |  |
| `duration` | `IntFilter` |  |
| `parkingSpaceUtilizationEventRemoteId` | `BigIntFilter` |  |
| `occupancyPeriod` | `DatetimeRangeFilter` |  |
| `occupancyDuration` | `IntFilter` |  |
| `occupancyStartParkingSpaceDataPointId` | `BigIntFilter` |  |
| `parkingSpaceDataPointsBySiteIdAndOccupancyStartParkingSpaceDataPointId` | `ParkingSpaceDataPointToManyParkingSpaceDataPointFilter` |  |
| `parkingSpaceDataPointsBySiteIdAndOccupancyStartParkingSpaceDataPointIdExist` | `Boolean` |  |
| `automationRuleExecutionDwellTimesBySiteIdAndParkingSpaceDataPointIdExist` | `Boolean` |  |
| `siteParkingSpaceUtilizationEventRemoteExists` | `Boolean` |  |
| `siteOccupancyStartParkingSpaceDataPointExists` | `Boolean` |  |
| `endTimestampUtc` | `DatetimeFilter` |  |

## ParkingSpaceDataPointHavingAverageInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `parkingSpaceId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `occupancyStatus` | `HavingIntFilter` |  |
| `duration` | `HavingIntFilter` |  |
| `parkingSpaceUtilizationEventRemoteId` | `HavingBigintFilter` |  |
| `occupancyDuration` | `HavingIntFilter` |  |
| `occupancyStartParkingSpaceDataPointId` | `HavingBigintFilter` |  |

## ParkingSpaceDataPointHavingDistinctCountInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `parkingSpaceId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `occupancyStatus` | `HavingIntFilter` |  |
| `duration` | `HavingIntFilter` |  |
| `parkingSpaceUtilizationEventRemoteId` | `HavingBigintFilter` |  |
| `occupancyDuration` | `HavingIntFilter` |  |
| `occupancyStartParkingSpaceDataPointId` | `HavingBigintFilter` |  |

## ParkingSpaceDataPointHavingInput (input-object)

Description Conditions for ParkingSpaceDataPoint aggregates.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `AND` | `[ParkingSpaceDataPointHavingInput!]` |  |
| `OR` | `[ParkingSpaceDataPointHavingInput!]` |  |
| `sum` | `ParkingSpaceDataPointHavingSumInput` |  |
| `distinctCount` | `ParkingSpaceDataPointHavingDistinctCountInput` |  |
| `min` | `ParkingSpaceDataPointHavingMinInput` |  |
| `max` | `ParkingSpaceDataPointHavingMaxInput` |  |
| `average` | `ParkingSpaceDataPointHavingAverageInput` |  |
| `stddevSample` | `ParkingSpaceDataPointHavingStddevSampleInput` |  |
| `stddevPopulation` | `ParkingSpaceDataPointHavingStddevPopulationInput` |  |
| `varianceSample` | `ParkingSpaceDataPointHavingVarianceSampleInput` |  |
| `variancePopulation` | `ParkingSpaceDataPointHavingVariancePopulationInput` |  |

## ParkingSpaceDataPointHavingMaxInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `parkingSpaceId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `occupancyStatus` | `HavingIntFilter` |  |
| `duration` | `HavingIntFilter` |  |
| `parkingSpaceUtilizationEventRemoteId` | `HavingBigintFilter` |  |
| `occupancyDuration` | `HavingIntFilter` |  |
| `occupancyStartParkingSpaceDataPointId` | `HavingBigintFilter` |  |

## ParkingSpaceDataPointHavingMinInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `parkingSpaceId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `occupancyStatus` | `HavingIntFilter` |  |
| `duration` | `HavingIntFilter` |  |
| `parkingSpaceUtilizationEventRemoteId` | `HavingBigintFilter` |  |
| `occupancyDuration` | `HavingIntFilter` |  |
| `occupancyStartParkingSpaceDataPointId` | `HavingBigintFilter` |  |

## ParkingSpaceDataPointHavingStddevPopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `parkingSpaceId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `occupancyStatus` | `HavingIntFilter` |  |
| `duration` | `HavingIntFilter` |  |
| `parkingSpaceUtilizationEventRemoteId` | `HavingBigintFilter` |  |
| `occupancyDuration` | `HavingIntFilter` |  |
| `occupancyStartParkingSpaceDataPointId` | `HavingBigintFilter` |  |

## ParkingSpaceDataPointHavingStddevSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `parkingSpaceId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `occupancyStatus` | `HavingIntFilter` |  |
| `duration` | `HavingIntFilter` |  |
| `parkingSpaceUtilizationEventRemoteId` | `HavingBigintFilter` |  |
| `occupancyDuration` | `HavingIntFilter` |  |
| `occupancyStartParkingSpaceDataPointId` | `HavingBigintFilter` |  |

## ParkingSpaceDataPointHavingSumInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `parkingSpaceId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `occupancyStatus` | `HavingIntFilter` |  |
| `duration` | `HavingIntFilter` |  |
| `parkingSpaceUtilizationEventRemoteId` | `HavingBigintFilter` |  |
| `occupancyDuration` | `HavingIntFilter` |  |
| `occupancyStartParkingSpaceDataPointId` | `HavingBigintFilter` |  |

## ParkingSpaceDataPointHavingVariancePopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `parkingSpaceId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `occupancyStatus` | `HavingIntFilter` |  |
| `duration` | `HavingIntFilter` |  |
| `parkingSpaceUtilizationEventRemoteId` | `HavingBigintFilter` |  |
| `occupancyDuration` | `HavingIntFilter` |  |
| `occupancyStartParkingSpaceDataPointId` | `HavingBigintFilter` |  |

## ParkingSpaceDataPointHavingVarianceSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `parkingSpaceId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `occupancyStatus` | `HavingIntFilter` |  |
| `duration` | `HavingIntFilter` |  |
| `parkingSpaceUtilizationEventRemoteId` | `HavingBigintFilter` |  |
| `occupancyDuration` | `HavingIntFilter` |  |
| `occupancyStartParkingSpaceDataPointId` | `HavingBigintFilter` |  |

## ParkingSpaceDataPointMaxAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `parkingSpaceId` | `IntFilter` |  |
| `siteLevelId` | `IntFilter` |  |
| `parkingAttributeId` | `IntFilter` |  |
| `siteId` | `IntFilter` |  |
| `occupancyStatus` | `IntFilter` |  |
| `duration` | `IntFilter` |  |
| `parkingSpaceUtilizationEventRemoteId` | `BigIntFilter` |  |
| `occupancyDuration` | `IntFilter` |  |
| `occupancyStartParkingSpaceDataPointId` | `BigIntFilter` |  |

## ParkingSpaceDataPointMinAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `parkingSpaceId` | `IntFilter` |  |
| `siteLevelId` | `IntFilter` |  |
| `parkingAttributeId` | `IntFilter` |  |
| `siteId` | `IntFilter` |  |
| `occupancyStatus` | `IntFilter` |  |
| `duration` | `IntFilter` |  |
| `parkingSpaceUtilizationEventRemoteId` | `BigIntFilter` |  |
| `occupancyDuration` | `IntFilter` |  |
| `occupancyStartParkingSpaceDataPointId` | `BigIntFilter` |  |

## ParkingSpaceDataPointStddevPopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `parkingSpaceId` | `BigFloatFilter` |  |
| `siteLevelId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `occupancyStatus` | `BigFloatFilter` |  |
| `duration` | `BigFloatFilter` |  |
| `parkingSpaceUtilizationEventRemoteId` | `BigFloatFilter` |  |
| `occupancyDuration` | `BigFloatFilter` |  |
| `occupancyStartParkingSpaceDataPointId` | `BigFloatFilter` |  |

## ParkingSpaceDataPointStddevSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `parkingSpaceId` | `BigFloatFilter` |  |
| `siteLevelId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `occupancyStatus` | `BigFloatFilter` |  |
| `duration` | `BigFloatFilter` |  |
| `parkingSpaceUtilizationEventRemoteId` | `BigFloatFilter` |  |
| `occupancyDuration` | `BigFloatFilter` |  |
| `occupancyStartParkingSpaceDataPointId` | `BigFloatFilter` |  |

## ParkingSpaceDataPointSumAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `parkingSpaceId` | `BigIntFilter` |  |
| `siteLevelId` | `BigIntFilter` |  |
| `parkingAttributeId` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `occupancyStatus` | `BigIntFilter` |  |
| `duration` | `BigIntFilter` |  |
| `parkingSpaceUtilizationEventRemoteId` | `BigFloatFilter` |  |
| `occupancyDuration` | `BigIntFilter` |  |
| `occupancyStartParkingSpaceDataPointId` | `BigFloatFilter` |  |

## ParkingSpaceDataPointToManyParkingSpaceDataPointFilter (input-object)

Description A filter to be used against many ParkingSpaceDataPoint object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingSpaceDataPointAggregatesFilter` |  |

## ParkingSpaceDataPointVariancePopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `parkingSpaceId` | `BigFloatFilter` |  |
| `siteLevelId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `occupancyStatus` | `BigFloatFilter` |  |
| `duration` | `BigFloatFilter` |  |
| `parkingSpaceUtilizationEventRemoteId` | `BigFloatFilter` |  |
| `occupancyDuration` | `BigFloatFilter` |  |
| `occupancyStartParkingSpaceDataPointId` | `BigFloatFilter` |  |

## ParkingSpaceDataPointVarianceSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `parkingSpaceId` | `BigFloatFilter` |  |
| `siteLevelId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `occupancyStatus` | `BigFloatFilter` |  |
| `duration` | `BigFloatFilter` |  |
| `parkingSpaceUtilizationEventRemoteId` | `BigFloatFilter` |  |
| `occupancyDuration` | `BigFloatFilter` |  |
| `occupancyStartParkingSpaceDataPointId` | `BigFloatFilter` |  |

## ParkingSpaceDataPointsConnection (object)

Description A connection to a list of ParkingSpaceDataPoint values.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodes` | `[ParkingSpaceDataPoint!]!` |  |
| `totalCount` | `Int!` |  |

## ParkingSpaceDataPointsOrderBy (enum)

Description Methods to use when ordering ParkingSpaceDataPoint .

## ParkingSpaceDistinctCountAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `siteLevelId` | `BigIntFilter` |  |
| `name` | `BigIntFilter` |  |
| `displayName` | `BigIntFilter` |  |
| `description` | `BigIntFilter` |  |
| `createdTimestamp` | `BigIntFilter` |  |
| `createdUserId` | `BigIntFilter` |  |
| `createdClientId` | `BigIntFilter` |  |
| `lastModifiedTimestamp` | `BigIntFilter` |  |
| `lastModifiedUserId` | `BigIntFilter` |  |
| `lastModifiedClientId` | `BigIntFilter` |  |
| `isDeleted` | `BigIntFilter` |  |
| `indicatedBySensorIdOld` | `BigIntFilter` |  |
| `detectedBySensorIdOld` | `BigIntFilter` |  |
| `isEnabled` | `BigIntFilter` |  |
| `indicatedBySensorId` | `BigIntFilter` |  |
| `detectedBySensorId` | `BigIntFilter` |  |

## ParkingSpaceFilter (input-object)

Description A filter to be used against ParkingSpace object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `siteId` | `IntFilter` |  |
| `siteLevelId` | `IntFilter` |  |
| `name` | `StringFilter` |  |
| `displayName` | `StringFilter` |  |
| `description` | `StringFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |
| `isDeleted` | `BooleanFilter` |  |
| `indicatedBySensorIdOld` | `StringFilter` |  |
| `detectedBySensorIdOld` | `StringFilter` |  |
| `isEnabled` | `BooleanFilter` |  |
| `indicatedBySensorId` | `BigIntFilter` |  |
| `detectedBySensorId` | `BigIntFilter` |  |
| `parkingSpaceUtilizationEventsExist` | `Boolean` |  |
| `snapshotParkingSpacesExist` | `Boolean` |  |
| `parkingSpaceDataPoints` | `ParkingSpaceToManyParkingSpaceDataPointFilter` |  |
| `parkingSpaceDataPointsExist` | `Boolean` |  |
| `parkingSpaceVehicleSessions` | `ParkingSpaceToManyParkingSpaceVehicleSessionFilter` |  |
| `parkingSpaceVehicleSessionsExist` | `Boolean` |  |
| `sensorParkingSpaceConfigurationExists` | `Boolean` |  |
| `createdUserExists` | `Boolean` |  |
| `lastModifiedUserExists` | `Boolean` |  |
| `indicatedBySensorExists` | `Boolean` |  |
| `detectedBySensorExists` | `Boolean` |  |

## ParkingSpaceHavingAverageInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `indicatedBySensorId` | `HavingBigintFilter` |  |
| `detectedBySensorId` | `HavingBigintFilter` |  |

## ParkingSpaceHavingDistinctCountInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `indicatedBySensorId` | `HavingBigintFilter` |  |
| `detectedBySensorId` | `HavingBigintFilter` |  |

## ParkingSpaceHavingInput (input-object)

Description Conditions for ParkingSpace aggregates.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `AND` | `[ParkingSpaceHavingInput!]` |  |
| `OR` | `[ParkingSpaceHavingInput!]` |  |
| `sum` | `ParkingSpaceHavingSumInput` |  |
| `distinctCount` | `ParkingSpaceHavingDistinctCountInput` |  |
| `min` | `ParkingSpaceHavingMinInput` |  |
| `max` | `ParkingSpaceHavingMaxInput` |  |
| `average` | `ParkingSpaceHavingAverageInput` |  |
| `stddevSample` | `ParkingSpaceHavingStddevSampleInput` |  |
| `stddevPopulation` | `ParkingSpaceHavingStddevPopulationInput` |  |
| `varianceSample` | `ParkingSpaceHavingVarianceSampleInput` |  |
| `variancePopulation` | `ParkingSpaceHavingVariancePopulationInput` |  |

## ParkingSpaceHavingMaxInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `indicatedBySensorId` | `HavingBigintFilter` |  |
| `detectedBySensorId` | `HavingBigintFilter` |  |

## ParkingSpaceHavingMinInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `indicatedBySensorId` | `HavingBigintFilter` |  |
| `detectedBySensorId` | `HavingBigintFilter` |  |

## ParkingSpaceHavingStddevPopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `indicatedBySensorId` | `HavingBigintFilter` |  |
| `detectedBySensorId` | `HavingBigintFilter` |  |

## ParkingSpaceHavingStddevSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `indicatedBySensorId` | `HavingBigintFilter` |  |
| `detectedBySensorId` | `HavingBigintFilter` |  |

## ParkingSpaceHavingSumInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `indicatedBySensorId` | `HavingBigintFilter` |  |
| `detectedBySensorId` | `HavingBigintFilter` |  |

## ParkingSpaceHavingVariancePopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `indicatedBySensorId` | `HavingBigintFilter` |  |
| `detectedBySensorId` | `HavingBigintFilter` |  |

## ParkingSpaceHavingVarianceSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `indicatedBySensorId` | `HavingBigintFilter` |  |
| `detectedBySensorId` | `HavingBigintFilter` |  |

## ParkingSpaceMaxAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `siteId` | `IntFilter` |  |
| `siteLevelId` | `IntFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |
| `indicatedBySensorId` | `BigIntFilter` |  |
| `detectedBySensorId` | `BigIntFilter` |  |

## ParkingSpaceMinAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `siteId` | `IntFilter` |  |
| `siteLevelId` | `IntFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |
| `indicatedBySensorId` | `BigIntFilter` |  |
| `detectedBySensorId` | `BigIntFilter` |  |

## ParkingSpaceStddevPopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `siteLevelId` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |
| `indicatedBySensorId` | `BigFloatFilter` |  |
| `detectedBySensorId` | `BigFloatFilter` |  |

## ParkingSpaceStddevSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `siteLevelId` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |
| `indicatedBySensorId` | `BigFloatFilter` |  |
| `detectedBySensorId` | `BigFloatFilter` |  |

## ParkingSpaceSumAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `siteLevelId` | `BigIntFilter` |  |
| `createdUserId` | `BigIntFilter` |  |
| `createdClientId` | `BigIntFilter` |  |
| `lastModifiedUserId` | `BigIntFilter` |  |
| `lastModifiedClientId` | `BigIntFilter` |  |
| `indicatedBySensorId` | `BigFloatFilter` |  |
| `detectedBySensorId` | `BigFloatFilter` |  |

## ParkingSpaceToManyParkingSpaceDataPointFilter (input-object)

Description A filter to be used against many ParkingSpaceDataPoint object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingSpaceDataPointAggregatesFilter` |  |

## ParkingSpaceToManyParkingSpaceVehicleSessionFilter (input-object)

Description A filter to be used against many ParkingSpaceVehicleSession object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingSpaceVehicleSessionAggregatesFilter` |  |

## ParkingSpaceVariancePopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `siteLevelId` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |
| `indicatedBySensorId` | `BigFloatFilter` |  |
| `detectedBySensorId` | `BigFloatFilter` |  |

## ParkingSpaceVarianceSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `siteLevelId` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |
| `indicatedBySensorId` | `BigFloatFilter` |  |
| `detectedBySensorId` | `BigFloatFilter` |  |

## ParkingSpaceVehicleSession (object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |
| `id` | `BigInt!` |  |
| `remoteId` | `Int!` |  |
| `parkingSpaceId` | `Int` |  |
| `isDeleted` | `Boolean!` |  |
| `remoteUpdatedAt` | `Datetime!` |  |
| `siteId` | `Int!` |  |
| `period` | `DatetimeRange!` |  |
| `duration` | `Int` |  |
| `modified` | `Boolean!` |  |
| `siteVehicleSessionId` | `BigInt` |  |
| `siteVehicleSessionMatchConfidence` | `BigFloat` |  |
| `remoteSiteVehicleSessionId` | `BigInt` |  |
| `remoteChangeSeqId` | `BigInt` |  |
| `updatedAt` | `Datetime!` |  |
| `parkingSpace` | `ParkingSpace` |  |
| `site` | `Site` |  |
| `parkingSpaceVehicleSessionVehicleRecognitions` | `ParkingSpaceVehicleSessionVehicleRecognitionsConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingSpaceVehicleSessionVehicleRecognitionsOrderBy!]: [ParkingSpaceVehicleSessionVehicleRecognitionsOrderBy!]`, `condition - ParkingSpaceVehicleSessionVehicleRecognitionCondition: ParkingSpaceVehicleSessionVehicleRecognitionCondition`, `filter - ParkingSpaceVehicleSessionVehicleRecognitionFilter: ParkingSpaceVehicleSessionVehicleRecognitionFilter` |

## ParkingSpaceVehicleSessionAggregatesFilter (input-object)

Description A filter to be used against aggregates of ParkingSpaceVehicleSession object types.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `sum` | `ParkingSpaceVehicleSessionSumAggregateFilter` |  |
| `distinctCount` | `ParkingSpaceVehicleSessionDistinctCountAggregateFilter` |  |
| `min` | `ParkingSpaceVehicleSessionMinAggregateFilter` |  |
| `max` | `ParkingSpaceVehicleSessionMaxAggregateFilter` |  |
| `average` | `ParkingSpaceVehicleSessionAverageAggregateFilter` |  |
| `stddevSample` | `ParkingSpaceVehicleSessionStddevSampleAggregateFilter` |  |
| `stddevPopulation` | `ParkingSpaceVehicleSessionStddevPopulationAggregateFilter` |  |
| `varianceSample` | `ParkingSpaceVehicleSessionVarianceSampleAggregateFilter` |  |
| `variancePopulation` | `ParkingSpaceVehicleSessionVariancePopulationAggregateFilter` |  |

## ParkingSpaceVehicleSessionAverageAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |
| `parkingSpaceId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `duration` | `BigFloatFilter` |  |
| `siteVehicleSessionId` | `BigFloatFilter` |  |
| `siteVehicleSessionMatchConfidence` | `BigFloatFilter` |  |
| `remoteSiteVehicleSessionId` | `BigFloatFilter` |  |
| `remoteChangeSeqId` | `BigFloatFilter` |  |

## ParkingSpaceVehicleSessionCondition (input-object)

Description A condition to be used against ParkingSpaceVehicleSession object types. All fields are tested for equality and combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigInt` |  |
| `remoteId` | `Int` |  |
| `parkingSpaceId` | `Int` |  |
| `isDeleted` | `Boolean` |  |
| `remoteUpdatedAt` | `Datetime` |  |
| `siteId` | `Int` |  |
| `period` | `DatetimeRangeInput` |  |
| `duration` | `Int` |  |
| `modified` | `Boolean` |  |
| `siteVehicleSessionId` | `BigInt` |  |
| `siteVehicleSessionMatchConfidence` | `BigFloat` |  |
| `remoteSiteVehicleSessionId` | `BigInt` |  |
| `remoteChangeSeqId` | `BigInt` |  |
| `updatedAt` | `Datetime` |  |

## ParkingSpaceVehicleSessionDistinctCountAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `remoteId` | `BigIntFilter` |  |
| `parkingSpaceId` | `BigIntFilter` |  |
| `isDeleted` | `BigIntFilter` |  |
| `remoteUpdatedAt` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `period` | `BigIntFilter` |  |
| `duration` | `BigIntFilter` |  |
| `modified` | `BigIntFilter` |  |
| `siteVehicleSessionId` | `BigIntFilter` |  |
| `siteVehicleSessionMatchConfidence` | `BigIntFilter` |  |
| `remoteSiteVehicleSessionId` | `BigIntFilter` |  |
| `remoteChangeSeqId` | `BigIntFilter` |  |
| `updatedAt` | `BigIntFilter` |  |

## ParkingSpaceVehicleSessionFilter (input-object)

Description A filter to be used against ParkingSpaceVehicleSession object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `remoteId` | `IntFilter` |  |
| `parkingSpaceId` | `IntFilter` |  |
| `isDeleted` | `BooleanFilter` |  |
| `remoteUpdatedAt` | `DatetimeFilter` |  |
| `siteId` | `IntFilter` |  |
| `period` | `DatetimeRangeFilter` |  |
| `duration` | `IntFilter` |  |
| `modified` | `BooleanFilter` |  |
| `siteVehicleSessionId` | `BigIntFilter` |  |
| `siteVehicleSessionMatchConfidence` | `BigFloatFilter` |  |
| `remoteSiteVehicleSessionId` | `BigIntFilter` |  |
| `remoteChangeSeqId` | `BigIntFilter` |  |
| `updatedAt` | `DatetimeFilter` |  |
| `parkingSpaceVehicleSessionBestAttributeExists` | `Boolean` |  |
| `parkingSpaceVehicleSessionVehicleRecognitions` | `ParkingSpaceVehicleSessionToManyParkingSpaceVehicleSessionVehicleRecognitionFilter` |  |
| `parkingSpaceVehicleSessionVehicleRecognitionsExist` | `Boolean` |  |
| `parkingSpaceVehicleSessionPlatesExist` | `Boolean` |  |
| `parkingSpaceExists` | `Boolean` |  |
| `siteSiteVehicleSessionExists` | `Boolean` |  |

## ParkingSpaceVehicleSessionHavingAverageInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `parkingSpaceId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `duration` | `HavingIntFilter` |  |
| `siteVehicleSessionId` | `HavingBigintFilter` |  |
| `remoteSiteVehicleSessionId` | `HavingBigintFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## ParkingSpaceVehicleSessionHavingDistinctCountInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `parkingSpaceId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `duration` | `HavingIntFilter` |  |
| `siteVehicleSessionId` | `HavingBigintFilter` |  |
| `remoteSiteVehicleSessionId` | `HavingBigintFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## ParkingSpaceVehicleSessionHavingInput (input-object)

Description Conditions for ParkingSpaceVehicleSession aggregates.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `AND` | `[ParkingSpaceVehicleSessionHavingInput!]` |  |
| `OR` | `[ParkingSpaceVehicleSessionHavingInput!]` |  |
| `sum` | `ParkingSpaceVehicleSessionHavingSumInput` |  |
| `distinctCount` | `ParkingSpaceVehicleSessionHavingDistinctCountInput` |  |
| `min` | `ParkingSpaceVehicleSessionHavingMinInput` |  |
| `max` | `ParkingSpaceVehicleSessionHavingMaxInput` |  |
| `average` | `ParkingSpaceVehicleSessionHavingAverageInput` |  |
| `stddevSample` | `ParkingSpaceVehicleSessionHavingStddevSampleInput` |  |
| `stddevPopulation` | `ParkingSpaceVehicleSessionHavingStddevPopulationInput` |  |
| `varianceSample` | `ParkingSpaceVehicleSessionHavingVarianceSampleInput` |  |
| `variancePopulation` | `ParkingSpaceVehicleSessionHavingVariancePopulationInput` |  |

## ParkingSpaceVehicleSessionHavingMaxInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `parkingSpaceId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `duration` | `HavingIntFilter` |  |
| `siteVehicleSessionId` | `HavingBigintFilter` |  |
| `remoteSiteVehicleSessionId` | `HavingBigintFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## ParkingSpaceVehicleSessionHavingMinInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `parkingSpaceId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `duration` | `HavingIntFilter` |  |
| `siteVehicleSessionId` | `HavingBigintFilter` |  |
| `remoteSiteVehicleSessionId` | `HavingBigintFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## ParkingSpaceVehicleSessionHavingStddevPopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `parkingSpaceId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `duration` | `HavingIntFilter` |  |
| `siteVehicleSessionId` | `HavingBigintFilter` |  |
| `remoteSiteVehicleSessionId` | `HavingBigintFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## ParkingSpaceVehicleSessionHavingStddevSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `parkingSpaceId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `duration` | `HavingIntFilter` |  |
| `siteVehicleSessionId` | `HavingBigintFilter` |  |
| `remoteSiteVehicleSessionId` | `HavingBigintFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## ParkingSpaceVehicleSessionHavingSumInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `parkingSpaceId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `duration` | `HavingIntFilter` |  |
| `siteVehicleSessionId` | `HavingBigintFilter` |  |
| `remoteSiteVehicleSessionId` | `HavingBigintFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## ParkingSpaceVehicleSessionHavingVariancePopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `parkingSpaceId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `duration` | `HavingIntFilter` |  |
| `siteVehicleSessionId` | `HavingBigintFilter` |  |
| `remoteSiteVehicleSessionId` | `HavingBigintFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## ParkingSpaceVehicleSessionHavingVarianceSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `parkingSpaceId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `duration` | `HavingIntFilter` |  |
| `siteVehicleSessionId` | `HavingBigintFilter` |  |
| `remoteSiteVehicleSessionId` | `HavingBigintFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## ParkingSpaceVehicleSessionMaxAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `remoteId` | `IntFilter` |  |
| `parkingSpaceId` | `IntFilter` |  |
| `remoteUpdatedAt` | `DatetimeFilter` |  |
| `siteId` | `IntFilter` |  |
| `duration` | `IntFilter` |  |
| `siteVehicleSessionId` | `BigIntFilter` |  |
| `siteVehicleSessionMatchConfidence` | `BigFloatFilter` |  |
| `remoteSiteVehicleSessionId` | `BigIntFilter` |  |
| `remoteChangeSeqId` | `BigIntFilter` |  |
| `updatedAt` | `DatetimeFilter` |  |

## ParkingSpaceVehicleSessionMinAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `remoteId` | `IntFilter` |  |
| `parkingSpaceId` | `IntFilter` |  |
| `remoteUpdatedAt` | `DatetimeFilter` |  |
| `siteId` | `IntFilter` |  |
| `duration` | `IntFilter` |  |
| `siteVehicleSessionId` | `BigIntFilter` |  |
| `siteVehicleSessionMatchConfidence` | `BigFloatFilter` |  |
| `remoteSiteVehicleSessionId` | `BigIntFilter` |  |
| `remoteChangeSeqId` | `BigIntFilter` |  |
| `updatedAt` | `DatetimeFilter` |  |

## ParkingSpaceVehicleSessionStddevPopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |
| `parkingSpaceId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `duration` | `BigFloatFilter` |  |
| `siteVehicleSessionId` | `BigFloatFilter` |  |
| `siteVehicleSessionMatchConfidence` | `BigFloatFilter` |  |
| `remoteSiteVehicleSessionId` | `BigFloatFilter` |  |
| `remoteChangeSeqId` | `BigFloatFilter` |  |

## ParkingSpaceVehicleSessionStddevSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |
| `parkingSpaceId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `duration` | `BigFloatFilter` |  |
| `siteVehicleSessionId` | `BigFloatFilter` |  |
| `siteVehicleSessionMatchConfidence` | `BigFloatFilter` |  |
| `remoteSiteVehicleSessionId` | `BigFloatFilter` |  |
| `remoteChangeSeqId` | `BigFloatFilter` |  |

## ParkingSpaceVehicleSessionSumAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `remoteId` | `BigIntFilter` |  |
| `parkingSpaceId` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `duration` | `BigIntFilter` |  |
| `siteVehicleSessionId` | `BigFloatFilter` |  |
| `siteVehicleSessionMatchConfidence` | `BigFloatFilter` |  |
| `remoteSiteVehicleSessionId` | `BigFloatFilter` |  |
| `remoteChangeSeqId` | `BigFloatFilter` |  |

## ParkingSpaceVehicleSessionToManyParkingSpaceVehicleSessionVehicleRecognitionFilter (input-object)

Description A filter to be used against many ParkingSpaceVehicleSessionVehicleRecognition object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingSpaceVehicleSessionVehicleRecognitionAggregatesFilter` |  |

## ParkingSpaceVehicleSessionVariancePopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |
| `parkingSpaceId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `duration` | `BigFloatFilter` |  |
| `siteVehicleSessionId` | `BigFloatFilter` |  |
| `siteVehicleSessionMatchConfidence` | `BigFloatFilter` |  |
| `remoteSiteVehicleSessionId` | `BigFloatFilter` |  |
| `remoteChangeSeqId` | `BigFloatFilter` |  |

## ParkingSpaceVehicleSessionVarianceSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |
| `parkingSpaceId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `duration` | `BigFloatFilter` |  |
| `siteVehicleSessionId` | `BigFloatFilter` |  |
| `siteVehicleSessionMatchConfidence` | `BigFloatFilter` |  |
| `remoteSiteVehicleSessionId` | `BigFloatFilter` |  |
| `remoteChangeSeqId` | `BigFloatFilter` |  |

## ParkingSpaceVehicleSessionVehicleRecognition (object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |
| `id` | `BigInt!` |  |
| `remoteId` | `Int!` |  |
| `parkingSpaceVehicleSessionId` | `BigInt` |  |
| `vehicleRecognitionId` | `BigInt` |  |
| `remoteUpdatedAt` | `Datetime!` |  |
| `siteId` | `Int!` |  |
| `remoteChangeSeqId` | `BigInt` |  |
| `updatedAt` | `Datetime` |  |
| `site` | `Site` |  |
| `parkingSpaceVehicleSession` | `ParkingSpaceVehicleSession` |  |
| `vehicleRecognition` | `VehicleRecognition` |  |

## ParkingSpaceVehicleSessionVehicleRecognitionAggregatesFilter (input-object)

Description A filter to be used against aggregates of ParkingSpaceVehicleSessionVehicleRecognition object types.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `sum` | `ParkingSpaceVehicleSessionVehicleRecognitionSumAggregateFilter` |  |
| `distinctCount` | `ParkingSpaceVehicleSessionVehicleRecognitionDistinctCountAggregateFilter` |  |
| `min` | `ParkingSpaceVehicleSessionVehicleRecognitionMinAggregateFilter` |  |
| `max` | `ParkingSpaceVehicleSessionVehicleRecognitionMaxAggregateFilter` |  |
| `average` | `ParkingSpaceVehicleSessionVehicleRecognitionAverageAggregateFilter` |  |
| `stddevSample` | `ParkingSpaceVehicleSessionVehicleRecognitionStddevSampleAggregateFilter` |  |
| `stddevPopulation` | `ParkingSpaceVehicleSessionVehicleRecognitionStddevPopulationAggregateFilter` |  |
| `varianceSample` | `ParkingSpaceVehicleSessionVehicleRecognitionVarianceSampleAggregateFilter` |  |
| `variancePopulation` | `ParkingSpaceVehicleSessionVehicleRecognitionVariancePopulationAggregateFilter` |  |

## ParkingSpaceVehicleSessionVehicleRecognitionAverageAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |
| `parkingSpaceVehicleSessionId` | `BigFloatFilter` |  |
| `vehicleRecognitionId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `remoteChangeSeqId` | `BigFloatFilter` |  |

## ParkingSpaceVehicleSessionVehicleRecognitionCondition (input-object)

Description A condition to be used against ParkingSpaceVehicleSessionVehicleRecognition object types. All fields are tested for equality and combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigInt` |  |
| `remoteId` | `Int` |  |
| `parkingSpaceVehicleSessionId` | `BigInt` |  |
| `vehicleRecognitionId` | `BigInt` |  |
| `remoteUpdatedAt` | `Datetime` |  |
| `siteId` | `Int` |  |
| `remoteChangeSeqId` | `BigInt` |  |
| `updatedAt` | `Datetime` |  |

## ParkingSpaceVehicleSessionVehicleRecognitionDistinctCountAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `remoteId` | `BigIntFilter` |  |
| `parkingSpaceVehicleSessionId` | `BigIntFilter` |  |
| `vehicleRecognitionId` | `BigIntFilter` |  |
| `remoteUpdatedAt` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `remoteChangeSeqId` | `BigIntFilter` |  |
| `updatedAt` | `BigIntFilter` |  |

## ParkingSpaceVehicleSessionVehicleRecognitionFilter (input-object)

Description A filter to be used against ParkingSpaceVehicleSessionVehicleRecognition object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `remoteId` | `IntFilter` |  |
| `parkingSpaceVehicleSessionId` | `BigIntFilter` |  |
| `vehicleRecognitionId` | `BigIntFilter` |  |
| `remoteUpdatedAt` | `DatetimeFilter` |  |
| `siteId` | `IntFilter` |  |
| `remoteChangeSeqId` | `BigIntFilter` |  |
| `updatedAt` | `DatetimeFilter` |  |
| `parkingSpaceVehicleSessionExists` | `Boolean` |  |
| `vehicleRecognitionExists` | `Boolean` |  |

## ParkingSpaceVehicleSessionVehicleRecognitionHavingAverageInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `parkingSpaceVehicleSessionId` | `HavingBigintFilter` |  |
| `vehicleRecognitionId` | `HavingBigintFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## ParkingSpaceVehicleSessionVehicleRecognitionHavingDistinctCountInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `parkingSpaceVehicleSessionId` | `HavingBigintFilter` |  |
| `vehicleRecognitionId` | `HavingBigintFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## ParkingSpaceVehicleSessionVehicleRecognitionHavingInput (input-object)

Description Conditions for ParkingSpaceVehicleSessionVehicleRecognition aggregates.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `AND` | `[ParkingSpaceVehicleSessionVehicleRecognitionHavingInput!]` |  |
| `OR` | `[ParkingSpaceVehicleSessionVehicleRecognitionHavingInput!]` |  |
| `sum` | `ParkingSpaceVehicleSessionVehicleRecognitionHavingSumInput` |  |
| `distinctCount` | `ParkingSpaceVehicleSessionVehicleRecognitionHavingDistinctCountInput` |  |
| `min` | `ParkingSpaceVehicleSessionVehicleRecognitionHavingMinInput` |  |
| `max` | `ParkingSpaceVehicleSessionVehicleRecognitionHavingMaxInput` |  |
| `average` | `ParkingSpaceVehicleSessionVehicleRecognitionHavingAverageInput` |  |
| `stddevSample` | `ParkingSpaceVehicleSessionVehicleRecognitionHavingStddevSampleInput` |  |
| `stddevPopulation` | `ParkingSpaceVehicleSessionVehicleRecognitionHavingStddevPopulationInput` |  |
| `varianceSample` | `ParkingSpaceVehicleSessionVehicleRecognitionHavingVarianceSampleInput` |  |
| `variancePopulation` | `ParkingSpaceVehicleSessionVehicleRecognitionHavingVariancePopulationInput` |  |

## ParkingSpaceVehicleSessionVehicleRecognitionHavingMaxInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `parkingSpaceVehicleSessionId` | `HavingBigintFilter` |  |
| `vehicleRecognitionId` | `HavingBigintFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## ParkingSpaceVehicleSessionVehicleRecognitionHavingMinInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `parkingSpaceVehicleSessionId` | `HavingBigintFilter` |  |
| `vehicleRecognitionId` | `HavingBigintFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## ParkingSpaceVehicleSessionVehicleRecognitionHavingStddevPopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `parkingSpaceVehicleSessionId` | `HavingBigintFilter` |  |
| `vehicleRecognitionId` | `HavingBigintFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## ParkingSpaceVehicleSessionVehicleRecognitionHavingStddevSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `parkingSpaceVehicleSessionId` | `HavingBigintFilter` |  |
| `vehicleRecognitionId` | `HavingBigintFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## ParkingSpaceVehicleSessionVehicleRecognitionHavingSumInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `parkingSpaceVehicleSessionId` | `HavingBigintFilter` |  |
| `vehicleRecognitionId` | `HavingBigintFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## ParkingSpaceVehicleSessionVehicleRecognitionHavingVariancePopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `parkingSpaceVehicleSessionId` | `HavingBigintFilter` |  |
| `vehicleRecognitionId` | `HavingBigintFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## ParkingSpaceVehicleSessionVehicleRecognitionHavingVarianceSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `parkingSpaceVehicleSessionId` | `HavingBigintFilter` |  |
| `vehicleRecognitionId` | `HavingBigintFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## ParkingSpaceVehicleSessionVehicleRecognitionMaxAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `remoteId` | `IntFilter` |  |
| `parkingSpaceVehicleSessionId` | `BigIntFilter` |  |
| `vehicleRecognitionId` | `BigIntFilter` |  |
| `remoteUpdatedAt` | `DatetimeFilter` |  |
| `siteId` | `IntFilter` |  |
| `remoteChangeSeqId` | `BigIntFilter` |  |
| `updatedAt` | `DatetimeFilter` |  |

## ParkingSpaceVehicleSessionVehicleRecognitionMinAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `remoteId` | `IntFilter` |  |
| `parkingSpaceVehicleSessionId` | `BigIntFilter` |  |
| `vehicleRecognitionId` | `BigIntFilter` |  |
| `remoteUpdatedAt` | `DatetimeFilter` |  |
| `siteId` | `IntFilter` |  |
| `remoteChangeSeqId` | `BigIntFilter` |  |
| `updatedAt` | `DatetimeFilter` |  |

## ParkingSpaceVehicleSessionVehicleRecognitionStddevPopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |
| `parkingSpaceVehicleSessionId` | `BigFloatFilter` |  |
| `vehicleRecognitionId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `remoteChangeSeqId` | `BigFloatFilter` |  |

## ParkingSpaceVehicleSessionVehicleRecognitionStddevSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |
| `parkingSpaceVehicleSessionId` | `BigFloatFilter` |  |
| `vehicleRecognitionId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `remoteChangeSeqId` | `BigFloatFilter` |  |

## ParkingSpaceVehicleSessionVehicleRecognitionSumAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `remoteId` | `BigIntFilter` |  |
| `parkingSpaceVehicleSessionId` | `BigFloatFilter` |  |
| `vehicleRecognitionId` | `BigFloatFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `remoteChangeSeqId` | `BigFloatFilter` |  |

## ParkingSpaceVehicleSessionVehicleRecognitionVariancePopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |
| `parkingSpaceVehicleSessionId` | `BigFloatFilter` |  |
| `vehicleRecognitionId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `remoteChangeSeqId` | `BigFloatFilter` |  |

## ParkingSpaceVehicleSessionVehicleRecognitionVarianceSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |
| `parkingSpaceVehicleSessionId` | `BigFloatFilter` |  |
| `vehicleRecognitionId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `remoteChangeSeqId` | `BigFloatFilter` |  |

## ParkingSpaceVehicleSessionVehicleRecognitionsConnection (object)

Description A connection to a list of ParkingSpaceVehicleSessionVehicleRecognition values.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodes` | `[ParkingSpaceVehicleSessionVehicleRecognition!]!` |  |
| `totalCount` | `Int!` |  |

## ParkingSpaceVehicleSessionVehicleRecognitionsOrderBy (enum)

Description Methods to use when ordering ParkingSpaceVehicleSessionVehicleRecognition .

## ParkingSpaceVehicleSessionsConnection (object)

Description A connection to a list of ParkingSpaceVehicleSession values.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodes` | `[ParkingSpaceVehicleSession!]!` |  |
| `totalCount` | `Int!` |  |

## ParkingSpaceVehicleSessionsOrderBy (enum)

Description Methods to use when ordering ParkingSpaceVehicleSession .

## ParkingSpacesConnection (object)

Description A connection to a list of ParkingSpace values.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodes` | `[ParkingSpace!]!` |  |
| `totalCount` | `Int!` |  |

## ParkingSpacesOrderBy (enum)

Description Methods to use when ordering ParkingSpace .

## ParkingZone (object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |
| `id` | `Int!` |  |
| `siteId` | `Int!` |  |
| `siteLevelId` | `Int!` |  |
| `name` | `String!` |  |
| `displayName` | `String` |  |
| `description` | `String` |  |
| `createdTimestamp` | `Datetime!` |  |
| `createdUserId` | `Int` |  |
| `createdClientId` | `Int!` |  |
| `lastModifiedTimestamp` | `Datetime!` |  |
| `lastModifiedUserId` | `Int` |  |
| `lastModifiedClientId` | `Int!` |  |
| `isDeleted` | `Boolean!` |  |
| `indicatedBySensorIdOld` | `String` |  |
| `remoteId` | `Int` |  |
| `countedBySensorId` | `BigInt` |  |
| `indicatedBySensorId` | `BigInt` |  |
| `site` | `Site` |  |
| `siteLevel` | `SiteLevel` |  |
| `createdUser` | `User` |  |
| `lastModifiedUser` | `User` |  |
| `countedBySensor` | `Sensor` |  |
| `indicatedBySensor` | `Sensor` |  |
| `parkingZoneDataPoints` | `ParkingZoneDataPointsConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingZoneDataPointsOrderBy!]: [ParkingZoneDataPointsOrderBy!]`, `condition - ParkingZoneDataPointCondition: ParkingZoneDataPointCondition`, `filter - ParkingZoneDataPointFilter: ParkingZoneDataPointFilter` |
| `parkingZoneCounters` | `ParkingZoneCountersConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingZoneCountersOrderBy!]: [ParkingZoneCountersOrderBy!]`, `condition - ParkingZoneCounterCondition: ParkingZoneCounterCondition`, `filter - ParkingZoneCounterFilter: ParkingZoneCounterFilter` |

## ParkingZoneAggregatesFilter (input-object)

Description A filter to be used against aggregates of ParkingZone object types.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `sum` | `ParkingZoneSumAggregateFilter` |  |
| `distinctCount` | `ParkingZoneDistinctCountAggregateFilter` |  |
| `min` | `ParkingZoneMinAggregateFilter` |  |
| `max` | `ParkingZoneMaxAggregateFilter` |  |
| `average` | `ParkingZoneAverageAggregateFilter` |  |
| `stddevSample` | `ParkingZoneStddevSampleAggregateFilter` |  |
| `stddevPopulation` | `ParkingZoneStddevPopulationAggregateFilter` |  |
| `varianceSample` | `ParkingZoneVarianceSampleAggregateFilter` |  |
| `variancePopulation` | `ParkingZoneVariancePopulationAggregateFilter` |  |

## ParkingZoneAverageAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `siteLevelId` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |
| `countedBySensorId` | `BigFloatFilter` |  |
| `indicatedBySensorId` | `BigFloatFilter` |  |

## ParkingZoneCondition (input-object)

Description A condition to be used against ParkingZone object types. All fields are tested for equality and combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `Int` |  |
| `siteId` | `Int` |  |
| `siteLevelId` | `Int` |  |
| `name` | `String` |  |
| `displayName` | `String` |  |
| `description` | `String` |  |
| `createdTimestamp` | `Datetime` |  |
| `createdUserId` | `Int` |  |
| `createdClientId` | `Int` |  |
| `lastModifiedTimestamp` | `Datetime` |  |
| `lastModifiedUserId` | `Int` |  |
| `lastModifiedClientId` | `Int` |  |
| `isDeleted` | `Boolean` |  |
| `indicatedBySensorIdOld` | `String` |  |
| `remoteId` | `Int` |  |
| `countedBySensorId` | `BigInt` |  |
| `indicatedBySensorId` | `BigInt` |  |

## ParkingZoneCounter (object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |
| `id` | `Int!` |  |
| `entered` | `Int!` |  |
| `left` | `Int!` |  |
| `lastReset` | `Datetime` |  |
| `beginTimestamp` | `Datetime!` |  |
| `endTimestamp` | `Datetime` |  |
| `parkingZoneId` | `Int!` |  |
| `siteId` | `Int!` |  |
| `parkingZone` | `ParkingZone` |  |
| `site` | `Site` |  |

## ParkingZoneCounterAggregatesFilter (input-object)

Description A filter to be used against aggregates of ParkingZoneCounter object types.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `sum` | `ParkingZoneCounterSumAggregateFilter` |  |
| `distinctCount` | `ParkingZoneCounterDistinctCountAggregateFilter` |  |
| `min` | `ParkingZoneCounterMinAggregateFilter` |  |
| `max` | `ParkingZoneCounterMaxAggregateFilter` |  |
| `average` | `ParkingZoneCounterAverageAggregateFilter` |  |
| `stddevSample` | `ParkingZoneCounterStddevSampleAggregateFilter` |  |
| `stddevPopulation` | `ParkingZoneCounterStddevPopulationAggregateFilter` |  |
| `varianceSample` | `ParkingZoneCounterVarianceSampleAggregateFilter` |  |
| `variancePopulation` | `ParkingZoneCounterVariancePopulationAggregateFilter` |  |

## ParkingZoneCounterAverageAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `entered` | `BigFloatFilter` |  |
| `left` | `BigFloatFilter` |  |
| `parkingZoneId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |

## ParkingZoneCounterCondition (input-object)

Description A condition to be used against ParkingZoneCounter object types. All fields are tested for equality and combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `Int` |  |
| `entered` | `Int` |  |
| `left` | `Int` |  |
| `lastReset` | `Datetime` |  |
| `beginTimestamp` | `Datetime` |  |
| `endTimestamp` | `Datetime` |  |
| `parkingZoneId` | `Int` |  |
| `siteId` | `Int` |  |

## ParkingZoneCounterDistinctCountAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `entered` | `BigIntFilter` |  |
| `left` | `BigIntFilter` |  |
| `lastReset` | `BigIntFilter` |  |
| `beginTimestamp` | `BigIntFilter` |  |
| `endTimestamp` | `BigIntFilter` |  |
| `parkingZoneId` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |

## ParkingZoneCounterFilter (input-object)

Description A filter to be used against ParkingZoneCounter object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `entered` | `IntFilter` |  |
| `left` | `IntFilter` |  |
| `lastReset` | `DatetimeFilter` |  |
| `beginTimestamp` | `DatetimeFilter` |  |
| `endTimestamp` | `DatetimeFilter` |  |
| `parkingZoneId` | `IntFilter` |  |
| `siteId` | `IntFilter` |  |

## ParkingZoneCounterHavingAverageInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `entered` | `HavingIntFilter` |  |
| `left` | `HavingIntFilter` |  |
| `lastReset` | `HavingDatetimeFilter` |  |
| `beginTimestamp` | `HavingDatetimeFilter` |  |
| `endTimestamp` | `HavingDatetimeFilter` |  |
| `parkingZoneId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |

## ParkingZoneCounterHavingDistinctCountInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `entered` | `HavingIntFilter` |  |
| `left` | `HavingIntFilter` |  |
| `lastReset` | `HavingDatetimeFilter` |  |
| `beginTimestamp` | `HavingDatetimeFilter` |  |
| `endTimestamp` | `HavingDatetimeFilter` |  |
| `parkingZoneId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |

## ParkingZoneCounterHavingInput (input-object)

Description Conditions for ParkingZoneCounter aggregates.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `AND` | `[ParkingZoneCounterHavingInput!]` |  |
| `OR` | `[ParkingZoneCounterHavingInput!]` |  |
| `sum` | `ParkingZoneCounterHavingSumInput` |  |
| `distinctCount` | `ParkingZoneCounterHavingDistinctCountInput` |  |
| `min` | `ParkingZoneCounterHavingMinInput` |  |
| `max` | `ParkingZoneCounterHavingMaxInput` |  |
| `average` | `ParkingZoneCounterHavingAverageInput` |  |
| `stddevSample` | `ParkingZoneCounterHavingStddevSampleInput` |  |
| `stddevPopulation` | `ParkingZoneCounterHavingStddevPopulationInput` |  |
| `varianceSample` | `ParkingZoneCounterHavingVarianceSampleInput` |  |
| `variancePopulation` | `ParkingZoneCounterHavingVariancePopulationInput` |  |

## ParkingZoneCounterHavingMaxInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `entered` | `HavingIntFilter` |  |
| `left` | `HavingIntFilter` |  |
| `lastReset` | `HavingDatetimeFilter` |  |
| `beginTimestamp` | `HavingDatetimeFilter` |  |
| `endTimestamp` | `HavingDatetimeFilter` |  |
| `parkingZoneId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |

## ParkingZoneCounterHavingMinInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `entered` | `HavingIntFilter` |  |
| `left` | `HavingIntFilter` |  |
| `lastReset` | `HavingDatetimeFilter` |  |
| `beginTimestamp` | `HavingDatetimeFilter` |  |
| `endTimestamp` | `HavingDatetimeFilter` |  |
| `parkingZoneId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |

## ParkingZoneCounterHavingStddevPopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `entered` | `HavingIntFilter` |  |
| `left` | `HavingIntFilter` |  |
| `lastReset` | `HavingDatetimeFilter` |  |
| `beginTimestamp` | `HavingDatetimeFilter` |  |
| `endTimestamp` | `HavingDatetimeFilter` |  |
| `parkingZoneId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |

## ParkingZoneCounterHavingStddevSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `entered` | `HavingIntFilter` |  |
| `left` | `HavingIntFilter` |  |
| `lastReset` | `HavingDatetimeFilter` |  |
| `beginTimestamp` | `HavingDatetimeFilter` |  |
| `endTimestamp` | `HavingDatetimeFilter` |  |
| `parkingZoneId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |

## ParkingZoneCounterHavingSumInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `entered` | `HavingIntFilter` |  |
| `left` | `HavingIntFilter` |  |
| `lastReset` | `HavingDatetimeFilter` |  |
| `beginTimestamp` | `HavingDatetimeFilter` |  |
| `endTimestamp` | `HavingDatetimeFilter` |  |
| `parkingZoneId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |

## ParkingZoneCounterHavingVariancePopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `entered` | `HavingIntFilter` |  |
| `left` | `HavingIntFilter` |  |
| `lastReset` | `HavingDatetimeFilter` |  |
| `beginTimestamp` | `HavingDatetimeFilter` |  |
| `endTimestamp` | `HavingDatetimeFilter` |  |
| `parkingZoneId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |

## ParkingZoneCounterHavingVarianceSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `entered` | `HavingIntFilter` |  |
| `left` | `HavingIntFilter` |  |
| `lastReset` | `HavingDatetimeFilter` |  |
| `beginTimestamp` | `HavingDatetimeFilter` |  |
| `endTimestamp` | `HavingDatetimeFilter` |  |
| `parkingZoneId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |

## ParkingZoneCounterMaxAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `entered` | `IntFilter` |  |
| `left` | `IntFilter` |  |
| `lastReset` | `DatetimeFilter` |  |
| `beginTimestamp` | `DatetimeFilter` |  |
| `endTimestamp` | `DatetimeFilter` |  |
| `parkingZoneId` | `IntFilter` |  |
| `siteId` | `IntFilter` |  |

## ParkingZoneCounterMinAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `entered` | `IntFilter` |  |
| `left` | `IntFilter` |  |
| `lastReset` | `DatetimeFilter` |  |
| `beginTimestamp` | `DatetimeFilter` |  |
| `endTimestamp` | `DatetimeFilter` |  |
| `parkingZoneId` | `IntFilter` |  |
| `siteId` | `IntFilter` |  |

## ParkingZoneCounterStddevPopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `entered` | `BigFloatFilter` |  |
| `left` | `BigFloatFilter` |  |
| `parkingZoneId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |

## ParkingZoneCounterStddevSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `entered` | `BigFloatFilter` |  |
| `left` | `BigFloatFilter` |  |
| `parkingZoneId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |

## ParkingZoneCounterSumAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `entered` | `BigIntFilter` |  |
| `left` | `BigIntFilter` |  |
| `parkingZoneId` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |

## ParkingZoneCounterVariancePopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `entered` | `BigFloatFilter` |  |
| `left` | `BigFloatFilter` |  |
| `parkingZoneId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |

## ParkingZoneCounterVarianceSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `entered` | `BigFloatFilter` |  |
| `left` | `BigFloatFilter` |  |
| `parkingZoneId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |

## ParkingZoneCountersConnection (object)

Description A connection to a list of ParkingZoneCounter values.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodes` | `[ParkingZoneCounter!]!` |  |
| `totalCount` | `Int!` |  |

## ParkingZoneCountersOrderBy (enum)

Description Methods to use when ordering ParkingZoneCounter .

## ParkingZoneDataPoint (object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |
| `id` | `BigInt!` |  |
| `parkingZoneId` | `Int!` |  |
| `siteLevelId` | `Int!` |  |
| `parkingAttributeId` | `Int!` |  |
| `siteId` | `Int!` |  |
| `availableCount` | `Int!` |  |
| `totalCount` | `Int!` |  |
| `period` | `DatetimeRange!` |  |
| `parkingZoneUtilizationEventRemoteId` | `BigInt` |  |
| `availableCountDelta` | `Int` |  |
| `parkingZone` | `ParkingZone` |  |
| `siteLevel` | `SiteLevel` |  |
| `parkingAttribute` | `ParkingAttribute` |  |
| `site` | `Site` |  |

## ParkingZoneDataPointAggregatesFilter (input-object)

Description A filter to be used against aggregates of ParkingZoneDataPoint object types.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `sum` | `ParkingZoneDataPointSumAggregateFilter` |  |
| `distinctCount` | `ParkingZoneDataPointDistinctCountAggregateFilter` |  |
| `min` | `ParkingZoneDataPointMinAggregateFilter` |  |
| `max` | `ParkingZoneDataPointMaxAggregateFilter` |  |
| `average` | `ParkingZoneDataPointAverageAggregateFilter` |  |
| `stddevSample` | `ParkingZoneDataPointStddevSampleAggregateFilter` |  |
| `stddevPopulation` | `ParkingZoneDataPointStddevPopulationAggregateFilter` |  |
| `varianceSample` | `ParkingZoneDataPointVarianceSampleAggregateFilter` |  |
| `variancePopulation` | `ParkingZoneDataPointVariancePopulationAggregateFilter` |  |

## ParkingZoneDataPointAverageAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `parkingZoneId` | `BigFloatFilter` |  |
| `siteLevelId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `parkingZoneUtilizationEventRemoteId` | `BigFloatFilter` |  |
| `availableCountDelta` | `BigFloatFilter` |  |

## ParkingZoneDataPointCondition (input-object)

Description A condition to be used against ParkingZoneDataPoint object types. All fields are tested for equality and combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigInt` |  |
| `parkingZoneId` | `Int` |  |
| `siteLevelId` | `Int` |  |
| `parkingAttributeId` | `Int` |  |
| `siteId` | `Int` |  |
| `availableCount` | `Int` |  |
| `totalCount` | `Int` |  |
| `period` | `DatetimeRangeInput` |  |
| `parkingZoneUtilizationEventRemoteId` | `BigInt` |  |
| `availableCountDelta` | `Int` |  |

## ParkingZoneDataPointDistinctCountAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `parkingZoneId` | `BigIntFilter` |  |
| `siteLevelId` | `BigIntFilter` |  |
| `parkingAttributeId` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `availableCount` | `BigIntFilter` |  |
| `totalCount` | `BigIntFilter` |  |
| `period` | `BigIntFilter` |  |
| `parkingZoneUtilizationEventRemoteId` | `BigIntFilter` |  |
| `availableCountDelta` | `BigIntFilter` |  |

## ParkingZoneDataPointFilter (input-object)

Description A filter to be used against ParkingZoneDataPoint object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `parkingZoneId` | `IntFilter` |  |
| `siteLevelId` | `IntFilter` |  |
| `parkingAttributeId` | `IntFilter` |  |
| `siteId` | `IntFilter` |  |
| `availableCount` | `IntFilter` |  |
| `totalCount` | `IntFilter` |  |
| `period` | `DatetimeRangeFilter` |  |
| `parkingZoneUtilizationEventRemoteId` | `BigIntFilter` |  |
| `availableCountDelta` | `IntFilter` |  |
| `siteParkingZoneUtilizationEventRemoteExists` | `Boolean` |  |

## ParkingZoneDataPointHavingAverageInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `parkingZoneId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |
| `parkingZoneUtilizationEventRemoteId` | `HavingBigintFilter` |  |
| `availableCountDelta` | `HavingIntFilter` |  |

## ParkingZoneDataPointHavingDistinctCountInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `parkingZoneId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |
| `parkingZoneUtilizationEventRemoteId` | `HavingBigintFilter` |  |
| `availableCountDelta` | `HavingIntFilter` |  |

## ParkingZoneDataPointHavingInput (input-object)

Description Conditions for ParkingZoneDataPoint aggregates.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `AND` | `[ParkingZoneDataPointHavingInput!]` |  |
| `OR` | `[ParkingZoneDataPointHavingInput!]` |  |
| `sum` | `ParkingZoneDataPointHavingSumInput` |  |
| `distinctCount` | `ParkingZoneDataPointHavingDistinctCountInput` |  |
| `min` | `ParkingZoneDataPointHavingMinInput` |  |
| `max` | `ParkingZoneDataPointHavingMaxInput` |  |
| `average` | `ParkingZoneDataPointHavingAverageInput` |  |
| `stddevSample` | `ParkingZoneDataPointHavingStddevSampleInput` |  |
| `stddevPopulation` | `ParkingZoneDataPointHavingStddevPopulationInput` |  |
| `varianceSample` | `ParkingZoneDataPointHavingVarianceSampleInput` |  |
| `variancePopulation` | `ParkingZoneDataPointHavingVariancePopulationInput` |  |

## ParkingZoneDataPointHavingMaxInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `parkingZoneId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |
| `parkingZoneUtilizationEventRemoteId` | `HavingBigintFilter` |  |
| `availableCountDelta` | `HavingIntFilter` |  |

## ParkingZoneDataPointHavingMinInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `parkingZoneId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |
| `parkingZoneUtilizationEventRemoteId` | `HavingBigintFilter` |  |
| `availableCountDelta` | `HavingIntFilter` |  |

## ParkingZoneDataPointHavingStddevPopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `parkingZoneId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |
| `parkingZoneUtilizationEventRemoteId` | `HavingBigintFilter` |  |
| `availableCountDelta` | `HavingIntFilter` |  |

## ParkingZoneDataPointHavingStddevSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `parkingZoneId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |
| `parkingZoneUtilizationEventRemoteId` | `HavingBigintFilter` |  |
| `availableCountDelta` | `HavingIntFilter` |  |

## ParkingZoneDataPointHavingSumInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `parkingZoneId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |
| `parkingZoneUtilizationEventRemoteId` | `HavingBigintFilter` |  |
| `availableCountDelta` | `HavingIntFilter` |  |

## ParkingZoneDataPointHavingVariancePopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `parkingZoneId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |
| `parkingZoneUtilizationEventRemoteId` | `HavingBigintFilter` |  |
| `availableCountDelta` | `HavingIntFilter` |  |

## ParkingZoneDataPointHavingVarianceSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `parkingZoneId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `availableCount` | `HavingIntFilter` |  |
| `totalCount` | `HavingIntFilter` |  |
| `parkingZoneUtilizationEventRemoteId` | `HavingBigintFilter` |  |
| `availableCountDelta` | `HavingIntFilter` |  |

## ParkingZoneDataPointMaxAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `parkingZoneId` | `IntFilter` |  |
| `siteLevelId` | `IntFilter` |  |
| `parkingAttributeId` | `IntFilter` |  |
| `siteId` | `IntFilter` |  |
| `availableCount` | `IntFilter` |  |
| `totalCount` | `IntFilter` |  |
| `parkingZoneUtilizationEventRemoteId` | `BigIntFilter` |  |
| `availableCountDelta` | `IntFilter` |  |

## ParkingZoneDataPointMinAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `parkingZoneId` | `IntFilter` |  |
| `siteLevelId` | `IntFilter` |  |
| `parkingAttributeId` | `IntFilter` |  |
| `siteId` | `IntFilter` |  |
| `availableCount` | `IntFilter` |  |
| `totalCount` | `IntFilter` |  |
| `parkingZoneUtilizationEventRemoteId` | `BigIntFilter` |  |
| `availableCountDelta` | `IntFilter` |  |

## ParkingZoneDataPointStddevPopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `parkingZoneId` | `BigFloatFilter` |  |
| `siteLevelId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `parkingZoneUtilizationEventRemoteId` | `BigFloatFilter` |  |
| `availableCountDelta` | `BigFloatFilter` |  |

## ParkingZoneDataPointStddevSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `parkingZoneId` | `BigFloatFilter` |  |
| `siteLevelId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `parkingZoneUtilizationEventRemoteId` | `BigFloatFilter` |  |
| `availableCountDelta` | `BigFloatFilter` |  |

## ParkingZoneDataPointSumAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `parkingZoneId` | `BigIntFilter` |  |
| `siteLevelId` | `BigIntFilter` |  |
| `parkingAttributeId` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `availableCount` | `BigIntFilter` |  |
| `totalCount` | `BigIntFilter` |  |
| `parkingZoneUtilizationEventRemoteId` | `BigFloatFilter` |  |
| `availableCountDelta` | `BigIntFilter` |  |

## ParkingZoneDataPointVariancePopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `parkingZoneId` | `BigFloatFilter` |  |
| `siteLevelId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `parkingZoneUtilizationEventRemoteId` | `BigFloatFilter` |  |
| `availableCountDelta` | `BigFloatFilter` |  |

## ParkingZoneDataPointVarianceSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `parkingZoneId` | `BigFloatFilter` |  |
| `siteLevelId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `parkingZoneUtilizationEventRemoteId` | `BigFloatFilter` |  |
| `availableCountDelta` | `BigFloatFilter` |  |

## ParkingZoneDataPointsConnection (object)

Description A connection to a list of ParkingZoneDataPoint values.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodes` | `[ParkingZoneDataPoint!]!` |  |
| `totalCount` | `Int!` |  |

## ParkingZoneDataPointsOrderBy (enum)

Description Methods to use when ordering ParkingZoneDataPoint .

## ParkingZoneDistinctCountAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `siteLevelId` | `BigIntFilter` |  |
| `name` | `BigIntFilter` |  |
| `displayName` | `BigIntFilter` |  |
| `description` | `BigIntFilter` |  |
| `createdTimestamp` | `BigIntFilter` |  |
| `createdUserId` | `BigIntFilter` |  |
| `createdClientId` | `BigIntFilter` |  |
| `lastModifiedTimestamp` | `BigIntFilter` |  |
| `lastModifiedUserId` | `BigIntFilter` |  |
| `lastModifiedClientId` | `BigIntFilter` |  |
| `isDeleted` | `BigIntFilter` |  |
| `indicatedBySensorIdOld` | `BigIntFilter` |  |
| `remoteId` | `BigIntFilter` |  |
| `countedBySensorId` | `BigIntFilter` |  |
| `indicatedBySensorId` | `BigIntFilter` |  |

## ParkingZoneFilter (input-object)

Description A filter to be used against ParkingZone object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `siteId` | `IntFilter` |  |
| `siteLevelId` | `IntFilter` |  |
| `name` | `StringFilter` |  |
| `displayName` | `StringFilter` |  |
| `description` | `StringFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |
| `isDeleted` | `BooleanFilter` |  |
| `indicatedBySensorIdOld` | `StringFilter` |  |
| `remoteId` | `IntFilter` |  |
| `countedBySensorId` | `BigIntFilter` |  |
| `indicatedBySensorId` | `BigIntFilter` |  |
| `parkingZoneDataPoints` | `ParkingZoneToManyParkingZoneDataPointFilter` |  |
| `parkingZoneDataPointsExist` | `Boolean` |  |
| `parkingZoneUtilizationEventsExist` | `Boolean` |  |
| `parkingZoneCounters` | `ParkingZoneToManyParkingZoneCounterFilter` |  |
| `parkingZoneCountersExist` | `Boolean` |  |
| `camerasExist` | `Boolean` |  |
| `vehicleSessionsExist` | `Boolean` |  |
| `createdUserExists` | `Boolean` |  |
| `lastModifiedUserExists` | `Boolean` |  |
| `countedBySensorExists` | `Boolean` |  |
| `indicatedBySensorExists` | `Boolean` |  |

## ParkingZoneHavingAverageInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `countedBySensorId` | `HavingBigintFilter` |  |
| `indicatedBySensorId` | `HavingBigintFilter` |  |

## ParkingZoneHavingDistinctCountInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `countedBySensorId` | `HavingBigintFilter` |  |
| `indicatedBySensorId` | `HavingBigintFilter` |  |

## ParkingZoneHavingInput (input-object)

Description Conditions for ParkingZone aggregates.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `AND` | `[ParkingZoneHavingInput!]` |  |
| `OR` | `[ParkingZoneHavingInput!]` |  |
| `sum` | `ParkingZoneHavingSumInput` |  |
| `distinctCount` | `ParkingZoneHavingDistinctCountInput` |  |
| `min` | `ParkingZoneHavingMinInput` |  |
| `max` | `ParkingZoneHavingMaxInput` |  |
| `average` | `ParkingZoneHavingAverageInput` |  |
| `stddevSample` | `ParkingZoneHavingStddevSampleInput` |  |
| `stddevPopulation` | `ParkingZoneHavingStddevPopulationInput` |  |
| `varianceSample` | `ParkingZoneHavingVarianceSampleInput` |  |
| `variancePopulation` | `ParkingZoneHavingVariancePopulationInput` |  |

## ParkingZoneHavingMaxInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `countedBySensorId` | `HavingBigintFilter` |  |
| `indicatedBySensorId` | `HavingBigintFilter` |  |

## ParkingZoneHavingMinInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `countedBySensorId` | `HavingBigintFilter` |  |
| `indicatedBySensorId` | `HavingBigintFilter` |  |

## ParkingZoneHavingStddevPopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `countedBySensorId` | `HavingBigintFilter` |  |
| `indicatedBySensorId` | `HavingBigintFilter` |  |

## ParkingZoneHavingStddevSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `countedBySensorId` | `HavingBigintFilter` |  |
| `indicatedBySensorId` | `HavingBigintFilter` |  |

## ParkingZoneHavingSumInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `countedBySensorId` | `HavingBigintFilter` |  |
| `indicatedBySensorId` | `HavingBigintFilter` |  |

## ParkingZoneHavingVariancePopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `countedBySensorId` | `HavingBigintFilter` |  |
| `indicatedBySensorId` | `HavingBigintFilter` |  |

## ParkingZoneHavingVarianceSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `countedBySensorId` | `HavingBigintFilter` |  |
| `indicatedBySensorId` | `HavingBigintFilter` |  |

## ParkingZoneMaxAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `siteId` | `IntFilter` |  |
| `siteLevelId` | `IntFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |
| `remoteId` | `IntFilter` |  |
| `countedBySensorId` | `BigIntFilter` |  |
| `indicatedBySensorId` | `BigIntFilter` |  |

## ParkingZoneMinAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `siteId` | `IntFilter` |  |
| `siteLevelId` | `IntFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |
| `remoteId` | `IntFilter` |  |
| `countedBySensorId` | `BigIntFilter` |  |
| `indicatedBySensorId` | `BigIntFilter` |  |

## ParkingZoneStddevPopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `siteLevelId` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |
| `countedBySensorId` | `BigFloatFilter` |  |
| `indicatedBySensorId` | `BigFloatFilter` |  |

## ParkingZoneStddevSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `siteLevelId` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |
| `countedBySensorId` | `BigFloatFilter` |  |
| `indicatedBySensorId` | `BigFloatFilter` |  |

## ParkingZoneSumAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `siteLevelId` | `BigIntFilter` |  |
| `createdUserId` | `BigIntFilter` |  |
| `createdClientId` | `BigIntFilter` |  |
| `lastModifiedUserId` | `BigIntFilter` |  |
| `lastModifiedClientId` | `BigIntFilter` |  |
| `remoteId` | `BigIntFilter` |  |
| `countedBySensorId` | `BigFloatFilter` |  |
| `indicatedBySensorId` | `BigFloatFilter` |  |

## ParkingZoneToManyParkingZoneCounterFilter (input-object)

Description A filter to be used against many ParkingZoneCounter object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingZoneCounterAggregatesFilter` |  |

## ParkingZoneToManyParkingZoneDataPointFilter (input-object)

Description A filter to be used against many ParkingZoneDataPoint object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingZoneDataPointAggregatesFilter` |  |

## ParkingZoneVariancePopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `siteLevelId` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |
| `countedBySensorId` | `BigFloatFilter` |  |
| `indicatedBySensorId` | `BigFloatFilter` |  |

## ParkingZoneVarianceSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `siteLevelId` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |
| `countedBySensorId` | `BigFloatFilter` |  |
| `indicatedBySensorId` | `BigFloatFilter` |  |

## ParkingZonesConnection (object)

Description A connection to a list of ParkingZone values.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodes` | `[ParkingZone!]!` |  |
| `totalCount` | `Int!` |  |

## ParkingZonesOrderBy (enum)

Description Methods to use when ordering ParkingZone .

## ReportingSiteLevelParkingAttributeUsageByHour (object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `Int` |  |
| `parkingAttributeId` | `Int` |  |
| `hour` | `Datetime` |  |
| `siteId` | `Int` |  |
| `availableCount` | `BigFloat` |  |
| `noDataCount` | `BigFloat` |  |
| `occupiedCount` | `BigFloat` |  |
| `totalCount` | `BigFloat` |  |
| `finalizedAt` | `Datetime` |  |
| `site` | `Site` |  |
| `siteLevel` | `SiteLevel` |  |
| `parkingAttribute` | `ParkingAttribute` |  |

## ReportingSiteLevelParkingAttributeUsageByHourAggregatesFilter (input-object)

Description A filter to be used against aggregates of ReportingSiteLevelParkingAttributeUsageByHour object types.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `sum` | `ReportingSiteLevelParkingAttributeUsageByHourSumAggregateFilter` |  |
| `distinctCount` | `ReportingSiteLevelParkingAttributeUsageByHourDistinctCountAggregateFilter` |  |
| `min` | `ReportingSiteLevelParkingAttributeUsageByHourMinAggregateFilter` |  |
| `max` | `ReportingSiteLevelParkingAttributeUsageByHourMaxAggregateFilter` |  |
| `average` | `ReportingSiteLevelParkingAttributeUsageByHourAverageAggregateFilter` |  |
| `stddevSample` | `ReportingSiteLevelParkingAttributeUsageByHourStddevSampleAggregateFilter` |  |
| `stddevPopulation` | `ReportingSiteLevelParkingAttributeUsageByHourStddevPopulationAggregateFilter` |  |
| `varianceSample` | `ReportingSiteLevelParkingAttributeUsageByHourVarianceSampleAggregateFilter` |  |
| `variancePopulation` | `ReportingSiteLevelParkingAttributeUsageByHourVariancePopulationAggregateFilter` |  |

## ReportingSiteLevelParkingAttributeUsageByHourAverageAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |

## ReportingSiteLevelParkingAttributeUsageByHourCondition (input-object)

Description A condition to be used against ReportingSiteLevelParkingAttributeUsageByHour object types. All fields are tested for equality and combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `Int` |  |
| `parkingAttributeId` | `Int` |  |
| `hour` | `Datetime` |  |
| `siteId` | `Int` |  |
| `availableCount` | `BigFloat` |  |
| `noDataCount` | `BigFloat` |  |
| `occupiedCount` | `BigFloat` |  |
| `totalCount` | `BigFloat` |  |
| `finalizedAt` | `Datetime` |  |

## ReportingSiteLevelParkingAttributeUsageByHourDistinctCountAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `BigIntFilter` |  |
| `parkingAttributeId` | `BigIntFilter` |  |
| `hour` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `availableCount` | `BigIntFilter` |  |
| `noDataCount` | `BigIntFilter` |  |
| `occupiedCount` | `BigIntFilter` |  |
| `totalCount` | `BigIntFilter` |  |
| `finalizedAt` | `BigIntFilter` |  |

## ReportingSiteLevelParkingAttributeUsageByHourFilter (input-object)

Description A filter to be used against ReportingSiteLevelParkingAttributeUsageByHour object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `IntFilter` |  |
| `parkingAttributeId` | `IntFilter` |  |
| `hour` | `DatetimeFilter` |  |
| `siteId` | `IntFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `finalizedAt` | `DatetimeFilter` |  |
| `siteExists` | `Boolean` |  |
| `siteLevelExists` | `Boolean` |  |
| `parkingAttributeExists` | `Boolean` |  |

## ReportingSiteLevelParkingAttributeUsageByHourHavingAverageInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `hour` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `finalizedAt` | `HavingDatetimeFilter` |  |

## ReportingSiteLevelParkingAttributeUsageByHourHavingDistinctCountInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `hour` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `finalizedAt` | `HavingDatetimeFilter` |  |

## ReportingSiteLevelParkingAttributeUsageByHourHavingInput (input-object)

Description Conditions for ReportingSiteLevelParkingAttributeUsageByHour aggregates.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `AND` | `[ReportingSiteLevelParkingAttributeUsageByHourHavingInput!]` |  |
| `OR` | `[ReportingSiteLevelParkingAttributeUsageByHourHavingInput!]` |  |
| `sum` | `ReportingSiteLevelParkingAttributeUsageByHourHavingSumInput` |  |
| `distinctCount` | `ReportingSiteLevelParkingAttributeUsageByHourHavingDistinctCountInput` |  |
| `min` | `ReportingSiteLevelParkingAttributeUsageByHourHavingMinInput` |  |
| `max` | `ReportingSiteLevelParkingAttributeUsageByHourHavingMaxInput` |  |
| `average` | `ReportingSiteLevelParkingAttributeUsageByHourHavingAverageInput` |  |
| `stddevSample` | `ReportingSiteLevelParkingAttributeUsageByHourHavingStddevSampleInput` |  |
| `stddevPopulation` | `ReportingSiteLevelParkingAttributeUsageByHourHavingStddevPopulationInput` |  |
| `varianceSample` | `ReportingSiteLevelParkingAttributeUsageByHourHavingVarianceSampleInput` |  |
| `variancePopulation` | `ReportingSiteLevelParkingAttributeUsageByHourHavingVariancePopulationInput` |  |

## ReportingSiteLevelParkingAttributeUsageByHourHavingMaxInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `hour` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `finalizedAt` | `HavingDatetimeFilter` |  |

## ReportingSiteLevelParkingAttributeUsageByHourHavingMinInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `hour` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `finalizedAt` | `HavingDatetimeFilter` |  |

## ReportingSiteLevelParkingAttributeUsageByHourHavingStddevPopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `hour` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `finalizedAt` | `HavingDatetimeFilter` |  |

## ReportingSiteLevelParkingAttributeUsageByHourHavingStddevSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `hour` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `finalizedAt` | `HavingDatetimeFilter` |  |

## ReportingSiteLevelParkingAttributeUsageByHourHavingSumInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `hour` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `finalizedAt` | `HavingDatetimeFilter` |  |

## ReportingSiteLevelParkingAttributeUsageByHourHavingVariancePopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `hour` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `finalizedAt` | `HavingDatetimeFilter` |  |

## ReportingSiteLevelParkingAttributeUsageByHourHavingVarianceSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `hour` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `finalizedAt` | `HavingDatetimeFilter` |  |

## ReportingSiteLevelParkingAttributeUsageByHourMaxAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `IntFilter` |  |
| `parkingAttributeId` | `IntFilter` |  |
| `hour` | `DatetimeFilter` |  |
| `siteId` | `IntFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `finalizedAt` | `DatetimeFilter` |  |

## ReportingSiteLevelParkingAttributeUsageByHourMinAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `IntFilter` |  |
| `parkingAttributeId` | `IntFilter` |  |
| `hour` | `DatetimeFilter` |  |
| `siteId` | `IntFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `finalizedAt` | `DatetimeFilter` |  |

## ReportingSiteLevelParkingAttributeUsageByHourStddevPopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |

## ReportingSiteLevelParkingAttributeUsageByHourStddevSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |

## ReportingSiteLevelParkingAttributeUsageByHourSumAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `BigIntFilter` |  |
| `parkingAttributeId` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |

## ReportingSiteLevelParkingAttributeUsageByHourVariancePopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |

## ReportingSiteLevelParkingAttributeUsageByHourVarianceSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteLevelId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |

## ReportingSiteLevelParkingAttributeUsageByHoursConnection (object)

Description A connection to a list of ReportingSiteLevelParkingAttributeUsageByHour values.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodes` | `[ReportingSiteLevelParkingAttributeUsageByHour!]!` |  |
| `totalCount` | `Int!` |  |

## ReportingSiteLevelParkingAttributeUsageByHoursOrderBy (enum)

Description Methods to use when ordering ReportingSiteLevelParkingAttributeUsageByHour .

## Sensor (object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |
| `id` | `BigInt!` |  |
| `sensorId` | `String!` |  |
| `configurationName` | `String` |  |
| `configurationDescription` | `String` |  |
| `siteId` | `Int!` |  |
| `isDeleted` | `Boolean!` |  |
| `site` | `Site` |  |
| `parkingSpacesByIndicatedBySensor` | `ParkingSpacesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingSpacesOrderBy!]: [ParkingSpacesOrderBy!]`, `condition - ParkingSpaceCondition: ParkingSpaceCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - ParkingSpaceFilter: ParkingSpaceFilter` |
| `parkingSpacesByDetectedBySensor` | `ParkingSpacesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingSpacesOrderBy!]: [ParkingSpacesOrderBy!]`, `condition - ParkingSpaceCondition: ParkingSpaceCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - ParkingSpaceFilter: ParkingSpaceFilter` |
| `parkingZonesByCountedBySensor` | `ParkingZonesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingZonesOrderBy!]: [ParkingZonesOrderBy!]`, `condition - ParkingZoneCondition: ParkingZoneCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - ParkingZoneFilter: ParkingZoneFilter` |
| `parkingZonesByIndicatedBySensor` | `ParkingZonesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingZonesOrderBy!]: [ParkingZonesOrderBy!]`, `condition - ParkingZoneCondition: ParkingZoneCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - ParkingZoneFilter: ParkingZoneFilter` |

## SensorAggregatesFilter (input-object)

Description A filter to be used against aggregates of Sensor object types.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `sum` | `SensorSumAggregateFilter` |  |
| `distinctCount` | `SensorDistinctCountAggregateFilter` |  |
| `min` | `SensorMinAggregateFilter` |  |
| `max` | `SensorMaxAggregateFilter` |  |
| `average` | `SensorAverageAggregateFilter` |  |
| `stddevSample` | `SensorStddevSampleAggregateFilter` |  |
| `stddevPopulation` | `SensorStddevPopulationAggregateFilter` |  |
| `varianceSample` | `SensorVarianceSampleAggregateFilter` |  |
| `variancePopulation` | `SensorVariancePopulationAggregateFilter` |  |

## SensorAverageAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |

## SensorCondition (input-object)

Description A condition to be used against Sensor object types. All fields are tested for equality and combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigInt` |  |
| `sensorId` | `String` |  |
| `configurationName` | `String` |  |
| `configurationDescription` | `String` |  |
| `siteId` | `Int` |  |
| `isDeleted` | `Boolean` |  |

## SensorDistinctCountAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `sensorId` | `BigIntFilter` |  |
| `configurationName` | `BigIntFilter` |  |
| `configurationDescription` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `isDeleted` | `BigIntFilter` |  |

## SensorFilter (input-object)

Description A filter to be used against Sensor object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `sensorId` | `StringFilter` |  |
| `configurationName` | `StringFilter` |  |
| `configurationDescription` | `StringFilter` |  |
| `siteId` | `IntFilter` |  |
| `isDeleted` | `BooleanFilter` |  |
| `sensorConnectionHistoriesExist` | `Boolean` |  |
| `sensorStorageStatusHistoriesExist` | `Boolean` |  |
| `parkingSpacesByIndicatedBySensor` | `SensorToManyParkingSpaceFilter` |  |
| `parkingSpacesByIndicatedBySensorExist` | `Boolean` |  |
| `parkingSpacesByDetectedBySensor` | `SensorToManyParkingSpaceFilter` |  |
| `parkingSpacesByDetectedBySensorExist` | `Boolean` |  |
| `parkingZonesByCountedBySensor` | `SensorToManyParkingZoneFilter` |  |
| `parkingZonesByCountedBySensorExist` | `Boolean` |  |
| `parkingZonesByIndicatedBySensor` | `SensorToManyParkingZoneFilter` |  |
| `parkingZonesByIndicatedBySensorExist` | `Boolean` |  |
| `sensorRemoteConnectionExists` | `Boolean` |  |
| `sensorReplacementsByOldSensorIdExist` | `Boolean` |  |
| `sensorReplacementsByNewSensorIdExist` | `Boolean` |  |
| `sensorStatusHistoriesExist` | `Boolean` |  |
| `snapshotParkingSpacesByIndicatedBySensorIdExist` | `Boolean` |  |
| `snapshotParkingSpacesByDetectedBySensorIdExist` | `Boolean` |  |
| `sensorConfigurationsExist` | `Boolean` |  |
| `sensorParkingSpaceConfigurationsExist` | `Boolean` |  |

## SensorHavingAverageInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `siteId` | `HavingIntFilter` |  |

## SensorHavingDistinctCountInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `siteId` | `HavingIntFilter` |  |

## SensorHavingInput (input-object)

Description Conditions for Sensor aggregates.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `AND` | `[SensorHavingInput!]` |  |
| `OR` | `[SensorHavingInput!]` |  |
| `sum` | `SensorHavingSumInput` |  |
| `distinctCount` | `SensorHavingDistinctCountInput` |  |
| `min` | `SensorHavingMinInput` |  |
| `max` | `SensorHavingMaxInput` |  |
| `average` | `SensorHavingAverageInput` |  |
| `stddevSample` | `SensorHavingStddevSampleInput` |  |
| `stddevPopulation` | `SensorHavingStddevPopulationInput` |  |
| `varianceSample` | `SensorHavingVarianceSampleInput` |  |
| `variancePopulation` | `SensorHavingVariancePopulationInput` |  |

## SensorHavingMaxInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `siteId` | `HavingIntFilter` |  |

## SensorHavingMinInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `siteId` | `HavingIntFilter` |  |

## SensorHavingStddevPopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `siteId` | `HavingIntFilter` |  |

## SensorHavingStddevSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `siteId` | `HavingIntFilter` |  |

## SensorHavingSumInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `siteId` | `HavingIntFilter` |  |

## SensorHavingVariancePopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `siteId` | `HavingIntFilter` |  |

## SensorHavingVarianceSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `siteId` | `HavingIntFilter` |  |

## SensorMaxAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `siteId` | `IntFilter` |  |

## SensorMinAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `siteId` | `IntFilter` |  |

## SensorStddevPopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |

## SensorStddevSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |

## SensorSumAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigIntFilter` |  |

## SensorToManyParkingSpaceFilter (input-object)

Description A filter to be used against many ParkingSpace object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingSpaceAggregatesFilter` |  |

## SensorToManyParkingZoneFilter (input-object)

Description A filter to be used against many ParkingZone object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingZoneAggregatesFilter` |  |

## SensorVariancePopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |

## SensorVarianceSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |

## SensorsConnection (object)

Description A connection to a list of Sensor values.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodes` | `[Sensor!]!` |  |
| `totalCount` | `Int!` |  |

## SensorsOrderBy (enum)

Description Methods to use when ordering Sensor .

## Site (object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |
| `id` | `Int!` |  |
| `siteUuid` | `String!` |  |
| `name` | `String!` |  |
| `displayName` | `String` |  |
| `description` | `String` |  |
| `address1` | `String` |  |
| `address2` | `String` |  |
| `address3` | `String` |  |
| `directions` | `String` |  |
| `timeZoneName` | `String` |  |
| `organizationId` | `Int!` |  |
| `isDataPollingActive` | `Boolean!` |  |
| `createdTimestamp` | `Datetime!` |  |
| `createdUserId` | `Int` |  |
| `createdClientId` | `Int!` |  |
| `lastModifiedTimestamp` | `Datetime!` |  |
| `lastModifiedUserId` | `Int` |  |
| `lastModifiedClientId` | `Int!` |  |
| `isDeleted` | `Boolean!` |  |
| `indicatorLightsEnabled` | `Boolean!` |  |
| `perimeterVehicleTrackingEnabled` | `Boolean!` |  |
| `vaultId` | `String` |  |
| `parkingSpaceVehicleTrackingEnabled` | `Boolean!` |  |
| `guidanceDisabledLedRgbaValue` | `Int` |  |
| `guidanceDisabledDisplayRgbaValue` | `Int` |  |
| `guidanceUnavailableLedRgbaValue` | `Int` |  |
| `guidanceUnavailableDisplayRgbaValue` | `Int` |  |
| `organization` | `Organization` |  |
| `createdUser` | `User` |  |
| `lastModifiedUser` | `User` |  |
| `parkingZoneDataPoints` | `ParkingZoneDataPointsConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingZoneDataPointsOrderBy!]: [ParkingZoneDataPointsOrderBy!]`, `condition - ParkingZoneDataPointCondition: ParkingZoneDataPointCondition`, `filter - ParkingZoneDataPointFilter: ParkingZoneDataPointFilter` |
| `siteLevelParkingAttributeUsageByHours` | `SiteLevelParkingAttributeUsageByHoursConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [SiteLevelParkingAttributeUsageByHoursOrderBy!]: [SiteLevelParkingAttributeUsageByHoursOrderBy!]`, `condition - SiteLevelParkingAttributeUsageByHourCondition: SiteLevelParkingAttributeUsageByHourCondition`, `filter - SiteLevelParkingAttributeUsageByHourFilter: SiteLevelParkingAttributeUsageByHourFilter` |
| `parkingZoneCounters` | `ParkingZoneCountersConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingZoneCountersOrderBy!]: [ParkingZoneCountersOrderBy!]`, `condition - ParkingZoneCounterCondition: ParkingZoneCounterCondition`, `filter - ParkingZoneCounterFilter: ParkingZoneCounterFilter` |
| `parkingAttributes` | `ParkingAttributesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingAttributesOrderBy!]: [ParkingAttributesOrderBy!]`, `condition - ParkingAttributeCondition: ParkingAttributeCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - ParkingAttributeFilter: ParkingAttributeFilter` |
| `parkingSpaces` | `ParkingSpacesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingSpacesOrderBy!]: [ParkingSpacesOrderBy!]`, `condition - ParkingSpaceCondition: ParkingSpaceCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - ParkingSpaceFilter: ParkingSpaceFilter` |
| `parkingZones` | `ParkingZonesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingZonesOrderBy!]: [ParkingZonesOrderBy!]`, `condition - ParkingZoneCondition: ParkingZoneCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - ParkingZoneFilter: ParkingZoneFilter` |
| `siteLevels` | `SiteLevelsConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [SiteLevelsOrderBy!]: [SiteLevelsOrderBy!]`, `condition - SiteLevelCondition: SiteLevelCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - SiteLevelFilter: SiteLevelFilter` |
| `sensors` | `SensorsConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [SensorsOrderBy!]: [SensorsOrderBy!]`, `condition - SensorCondition: SensorCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - SensorFilter: SensorFilter` |
| `parkingSpaceDataPoints` | `ParkingSpaceDataPointsConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingSpaceDataPointsOrderBy!]: [ParkingSpaceDataPointsOrderBy!]`, `condition - ParkingSpaceDataPointCondition: ParkingSpaceDataPointCondition`, `filter - ParkingSpaceDataPointFilter: ParkingSpaceDataPointFilter` |
| `vehicleRecognitions` | `VehicleRecognitionsConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [VehicleRecognitionsOrderBy!]: [VehicleRecognitionsOrderBy!]`, `condition - VehicleRecognitionCondition: VehicleRecognitionCondition`, `filter - VehicleRecognitionFilter: VehicleRecognitionFilter` |
| `vehicleRecognitionPlates` | `VehicleRecognitionPlatesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [VehicleRecognitionPlatesOrderBy!]: [VehicleRecognitionPlatesOrderBy!]`, `condition - VehicleRecognitionPlateCondition: VehicleRecognitionPlateCondition`, `filter - VehicleRecognitionPlateFilter: VehicleRecognitionPlateFilter` |
| `parkingSpaceVehicleSessions` | `ParkingSpaceVehicleSessionsConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingSpaceVehicleSessionsOrderBy!]: [ParkingSpaceVehicleSessionsOrderBy!]`, `condition - ParkingSpaceVehicleSessionCondition: ParkingSpaceVehicleSessionCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - ParkingSpaceVehicleSessionFilter: ParkingSpaceVehicleSessionFilter` |
| `parkingSpaceVehicleSessionVehicleRecognitions` | `ParkingSpaceVehicleSessionVehicleRecognitionsConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingSpaceVehicleSessionVehicleRecognitionsOrderBy!]: [ParkingSpaceVehicleSessionVehicleRecognitionsOrderBy!]`, `condition - ParkingSpaceVehicleSessionVehicleRecognitionCondition: ParkingSpaceVehicleSessionVehicleRecognitionCondition`, `filter - ParkingSpaceVehicleSessionVehicleRecognitionFilter: ParkingSpaceVehicleSessionVehicleRecognitionFilter` |
| `latestSiteLevelParkingAttributeParkingUsages` | `LatestSiteLevelParkingAttributeParkingUsagesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [LatestSiteLevelParkingAttributeParkingUsagesOrderBy!]: [LatestSiteLevelParkingAttributeParkingUsagesOrderBy!]`, `condition - LatestSiteLevelParkingAttributeParkingUsageCondition: LatestSiteLevelParkingAttributeParkingUsageCondition`, `filter - LatestSiteLevelParkingAttributeParkingUsageFilter: LatestSiteLevelParkingAttributeParkingUsageFilter` |
| `latestSiteParkingAttributeParkingUsages` | `LatestSiteParkingAttributeParkingUsagesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [LatestSiteParkingAttributeParkingUsagesOrderBy!]: [LatestSiteParkingAttributeParkingUsagesOrderBy!]`, `condition - LatestSiteParkingAttributeParkingUsageCondition: LatestSiteParkingAttributeParkingUsageCondition`, `filter - LatestSiteParkingAttributeParkingUsageFilter: LatestSiteParkingAttributeParkingUsageFilter` |
| `latestSiteParkingUsages` | `LatestSiteParkingUsagesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [LatestSiteParkingUsagesOrderBy!]: [LatestSiteParkingUsagesOrderBy!]`, `condition - LatestSiteParkingUsageCondition: LatestSiteParkingUsageCondition`, `filter - LatestSiteParkingUsageFilter: LatestSiteParkingUsageFilter` |
| `reportingSiteLevelParkingAttributeUsageByHours` | `ReportingSiteLevelParkingAttributeUsageByHoursConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ReportingSiteLevelParkingAttributeUsageByHoursOrderBy!]: [ReportingSiteLevelParkingAttributeUsageByHoursOrderBy!]`, `condition - ReportingSiteLevelParkingAttributeUsageByHourCondition: ReportingSiteLevelParkingAttributeUsageByHourCondition`, `filter - ReportingSiteLevelParkingAttributeUsageByHourFilter: ReportingSiteLevelParkingAttributeUsageByHourFilter` |

## SiteAggregatesFilter (input-object)

Description A filter to be used against aggregates of Site object types.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `sum` | `SiteSumAggregateFilter` |  |
| `distinctCount` | `SiteDistinctCountAggregateFilter` |  |
| `min` | `SiteMinAggregateFilter` |  |
| `max` | `SiteMaxAggregateFilter` |  |
| `average` | `SiteAverageAggregateFilter` |  |
| `stddevSample` | `SiteStddevSampleAggregateFilter` |  |
| `stddevPopulation` | `SiteStddevPopulationAggregateFilter` |  |
| `varianceSample` | `SiteVarianceSampleAggregateFilter` |  |
| `variancePopulation` | `SiteVariancePopulationAggregateFilter` |  |

## SiteAverageAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `organizationId` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |
| `guidanceDisabledLedRgbaValue` | `BigFloatFilter` |  |
| `guidanceDisabledDisplayRgbaValue` | `BigFloatFilter` |  |
| `guidanceUnavailableLedRgbaValue` | `BigFloatFilter` |  |
| `guidanceUnavailableDisplayRgbaValue` | `BigFloatFilter` |  |

## SiteCondition (input-object)

Description A condition to be used against Site object types. All fields are tested for equality and combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `Int` |  |
| `siteUuid` | `String` |  |
| `name` | `String` |  |
| `displayName` | `String` |  |
| `description` | `String` |  |
| `address1` | `String` |  |
| `address2` | `String` |  |
| `address3` | `String` |  |
| `directions` | `String` |  |
| `timeZoneName` | `String` |  |
| `organizationId` | `Int` |  |
| `isDataPollingActive` | `Boolean` |  |
| `createdTimestamp` | `Datetime` |  |
| `createdUserId` | `Int` |  |
| `createdClientId` | `Int` |  |
| `lastModifiedTimestamp` | `Datetime` |  |
| `lastModifiedUserId` | `Int` |  |
| `lastModifiedClientId` | `Int` |  |
| `isDeleted` | `Boolean` |  |
| `indicatorLightsEnabled` | `Boolean` |  |
| `perimeterVehicleTrackingEnabled` | `Boolean` |  |
| `vaultId` | `String` |  |
| `parkingSpaceVehicleTrackingEnabled` | `Boolean` |  |
| `guidanceDisabledLedRgbaValue` | `Int` |  |
| `guidanceDisabledDisplayRgbaValue` | `Int` |  |
| `guidanceUnavailableLedRgbaValue` | `Int` |  |
| `guidanceUnavailableDisplayRgbaValue` | `Int` |  |

## SiteDistinctCountAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `siteUuid` | `BigIntFilter` |  |
| `name` | `BigIntFilter` |  |
| `displayName` | `BigIntFilter` |  |
| `description` | `BigIntFilter` |  |
| `address1` | `BigIntFilter` |  |
| `address2` | `BigIntFilter` |  |
| `address3` | `BigIntFilter` |  |
| `directions` | `BigIntFilter` |  |
| `timeZoneName` | `BigIntFilter` |  |
| `organizationId` | `BigIntFilter` |  |
| `isDataPollingActive` | `BigIntFilter` |  |
| `createdTimestamp` | `BigIntFilter` |  |
| `createdUserId` | `BigIntFilter` |  |
| `createdClientId` | `BigIntFilter` |  |
| `lastModifiedTimestamp` | `BigIntFilter` |  |
| `lastModifiedUserId` | `BigIntFilter` |  |
| `lastModifiedClientId` | `BigIntFilter` |  |
| `isDeleted` | `BigIntFilter` |  |
| `indicatorLightsEnabled` | `BigIntFilter` |  |
| `perimeterVehicleTrackingEnabled` | `BigIntFilter` |  |
| `vaultId` | `BigIntFilter` |  |
| `parkingSpaceVehicleTrackingEnabled` | `BigIntFilter` |  |
| `guidanceDisabledLedRgbaValue` | `BigIntFilter` |  |
| `guidanceDisabledDisplayRgbaValue` | `BigIntFilter` |  |
| `guidanceUnavailableLedRgbaValue` | `BigIntFilter` |  |
| `guidanceUnavailableDisplayRgbaValue` | `BigIntFilter` |  |

## SiteFilter (input-object)

Description A filter to be used against Site object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `siteUuid` | `StringFilter` |  |
| `name` | `StringFilter` |  |
| `displayName` | `StringFilter` |  |
| `description` | `StringFilter` |  |
| `address1` | `StringFilter` |  |
| `address2` | `StringFilter` |  |
| `address3` | `StringFilter` |  |
| `directions` | `StringFilter` |  |
| `timeZoneName` | `StringFilter` |  |
| `organizationId` | `IntFilter` |  |
| `isDataPollingActive` | `BooleanFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |
| `isDeleted` | `BooleanFilter` |  |
| `indicatorLightsEnabled` | `BooleanFilter` |  |
| `perimeterVehicleTrackingEnabled` | `BooleanFilter` |  |
| `vaultId` | `StringFilter` |  |
| `parkingSpaceVehicleTrackingEnabled` | `BooleanFilter` |  |
| `guidanceDisabledLedRgbaValue` | `IntFilter` |  |
| `guidanceDisabledDisplayRgbaValue` | `IntFilter` |  |
| `guidanceUnavailableLedRgbaValue` | `IntFilter` |  |
| `guidanceUnavailableDisplayRgbaValue` | `IntFilter` |  |
| `parkingSpaceUtilizationEventsExist` | `Boolean` |  |
| `parkingZoneDataPoints` | `SiteToManyParkingZoneDataPointFilter` |  |
| `parkingZoneDataPointsExist` | `Boolean` |  |
| `parkingZoneUtilizationEventsExist` | `Boolean` |  |
| `siteLevelParkingAttributeUsageByHours` | `SiteToManySiteLevelParkingAttributeUsageByHourFilter` |  |
| `siteLevelParkingAttributeUsageByHoursExist` | `Boolean` |  |
| `gatewayExists` | `Boolean` |  |
| `gatewayConnectionHistoriesExist` | `Boolean` |  |
| `modemConnectionHistoriesExist` | `Boolean` |  |
| `networkAttachedStorageConnectionHistoriesExist` | `Boolean` |  |
| `networkDeviceStatusHistoriesExist` | `Boolean` |  |
| `parkingZoneCounters` | `SiteToManyParkingZoneCounterFilter` |  |
| `parkingZoneCountersExist` | `Boolean` |  |
| `sensorConnectionHistoriesExist` | `Boolean` |  |
| `sensorStorageStatusHistoriesExist` | `Boolean` |  |
| `signCounterHealthHistoriesExist` | `Boolean` |  |
| `camerasExist` | `Boolean` |  |
| `cameraConnectionHistoriesExist` | `Boolean` |  |
| `gatewayStatusHistoriesExist` | `Boolean` |  |
| `parkingAttributes` | `SiteToManyParkingAttributeFilter` |  |
| `parkingAttributesExist` | `Boolean` |  |
| `parkingSpaces` | `SiteToManyParkingSpaceFilter` |  |
| `parkingSpacesExist` | `Boolean` |  |
| `parkingZones` | `SiteToManyParkingZoneFilter` |  |
| `parkingZonesExist` | `Boolean` |  |
| `siteLevels` | `SiteToManySiteLevelFilter` |  |
| `siteLevelsExist` | `Boolean` |  |
| `lprProcessorsExist` | `Boolean` |  |
| `lprProcessorStatusHistoriesExist` | `Boolean` |  |
| `modemStatusHistoriesExist` | `Boolean` |  |
| `networkDevicesExist` | `Boolean` |  |
| `networkAttachedStoragesExist` | `Boolean` |  |
| `networkAttachedStorageConfigurationsExist` | `Boolean` |  |
| `networkControllerExists` | `Boolean` |  |
| `sensors` | `SiteToManySensorFilter` |  |
| `sensorsExist` | `Boolean` |  |
| `sensorRemoteConnectionsExist` | `Boolean` |  |
| `sensorReplacementsExist` | `Boolean` |  |
| `sensorStatusHistoriesExist` | `Boolean` |  |
| `signsExist` | `Boolean` |  |
| `signCountersExist` | `Boolean` |  |
| `snapshotsExist` | `Boolean` |  |
| `validationsExist` | `Boolean` |  |
| `notificationsExist` | `Boolean` |  |
| `modemsExist` | `Boolean` |  |
| `visitorKiosksExist` | `Boolean` |  |
| `visitorKioskAuthenticationsExist` | `Boolean` |  |
| `parkingSpaceDataPoints` | `SiteToManyParkingSpaceDataPointFilter` |  |
| `parkingSpaceDataPointsExist` | `Boolean` |  |
| `vehicleRecognitions` | `SiteToManyVehicleRecognitionFilter` |  |
| `vehicleRecognitionsExist` | `Boolean` |  |
| `vehicleRecognitionColorsExist` | `Boolean` |  |
| `vehicleRecognitionMakeModelsExist` | `Boolean` |  |
| `vehicleRecognitionOrientationsExist` | `Boolean` |  |
| `vehicleRecognitionPlates` | `SiteToManyVehicleRecognitionPlateFilter` |  |
| `vehicleRecognitionPlatesExist` | `Boolean` |  |
| `vehicleRecognitionPlateRegionsExist` | `Boolean` |  |
| `vehicleSessionsExist` | `Boolean` |  |
| `vehicleSessionBestAttributesExist` | `Boolean` |  |
| `parkingSpaceVehicleSessions` | `SiteToManyParkingSpaceVehicleSessionFilter` |  |
| `parkingSpaceVehicleSessionsExist` | `Boolean` |  |
| `parkingSpaceVehicleSessionBestAttributesExist` | `Boolean` |  |
| `parkingSpaceVehicleSessionVehicleRecognitions` | `SiteToManyParkingSpaceVehicleSessionVehicleRecognitionFilter` |  |
| `parkingSpaceVehicleSessionVehicleRecognitionsExist` | `Boolean` |  |
| `automationRulesExist` | `Boolean` |  |
| `automationRuleConditionPlateListsExist` | `Boolean` |  |
| `automationRuleConditionSiteLevelsExist` | `Boolean` |  |
| `automationRuleExecutionsExist` | `Boolean` |  |
| `automationRuleExecutionPlateRecognizedsExist` | `Boolean` |  |
| `automationRuleExecutionActionEmailsExist` | `Boolean` |  |
| `automationRuleActionEmailsExist` | `Boolean` |  |
| `automationRuleConditionDurationsExist` | `Boolean` |  |
| `automationRuleExecutionDwellTimesExist` | `Boolean` |  |
| `eventProcessingStatesExist` | `Boolean` |  |
| `lightingZonesExist` | `Boolean` |  |
| `sensorConfigurationsExist` | `Boolean` |  |
| `parkingSpaceVehicleSessionPlatesExist` | `Boolean` |  |
| `vehicleSessionPlatesExist` | `Boolean` |  |
| `unifiedVehicleSessionsExist` | `Boolean` |  |
| `sensorParkingSpaceConfigurationsExist` | `Boolean` |  |
| `latestSiteLevelParkingAttributeParkingUsages` | `SiteToManyLatestSiteLevelParkingAttributeParkingUsageFilter` |  |
| `latestSiteLevelParkingAttributeParkingUsagesExist` | `Boolean` |  |
| `latestSiteLevelParkingUsagesExist` | `Boolean` |  |
| `latestSiteParkingAttributeParkingUsages` | `SiteToManyLatestSiteParkingAttributeParkingUsageFilter` |  |
| `latestSiteParkingAttributeParkingUsagesExist` | `Boolean` |  |
| `latestSiteParkingUsages` | `SiteToManyLatestSiteParkingUsageFilter` |  |
| `latestSiteParkingUsagesExist` | `Boolean` |  |
| `reportingSiteLevelParkingAttributeUsageByHours` | `SiteToManyReportingSiteLevelParkingAttributeUsageByHourFilter` |  |
| `reportingSiteLevelParkingAttributeUsageByHoursExist` | `Boolean` |  |
| `sitePlateWithSessionCountsExist` | `Boolean` |  |
| `createdUserExists` | `Boolean` |  |
| `lastModifiedUserExists` | `Boolean` |  |

## SiteHavingAverageInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `organizationId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `guidanceDisabledLedRgbaValue` | `HavingIntFilter` |  |
| `guidanceDisabledDisplayRgbaValue` | `HavingIntFilter` |  |
| `guidanceUnavailableLedRgbaValue` | `HavingIntFilter` |  |
| `guidanceUnavailableDisplayRgbaValue` | `HavingIntFilter` |  |

## SiteHavingDistinctCountInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `organizationId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `guidanceDisabledLedRgbaValue` | `HavingIntFilter` |  |
| `guidanceDisabledDisplayRgbaValue` | `HavingIntFilter` |  |
| `guidanceUnavailableLedRgbaValue` | `HavingIntFilter` |  |
| `guidanceUnavailableDisplayRgbaValue` | `HavingIntFilter` |  |

## SiteHavingInput (input-object)

Description Conditions for Site aggregates.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `AND` | `[SiteHavingInput!]` |  |
| `OR` | `[SiteHavingInput!]` |  |
| `sum` | `SiteHavingSumInput` |  |
| `distinctCount` | `SiteHavingDistinctCountInput` |  |
| `min` | `SiteHavingMinInput` |  |
| `max` | `SiteHavingMaxInput` |  |
| `average` | `SiteHavingAverageInput` |  |
| `stddevSample` | `SiteHavingStddevSampleInput` |  |
| `stddevPopulation` | `SiteHavingStddevPopulationInput` |  |
| `varianceSample` | `SiteHavingVarianceSampleInput` |  |
| `variancePopulation` | `SiteHavingVariancePopulationInput` |  |

## SiteHavingMaxInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `organizationId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `guidanceDisabledLedRgbaValue` | `HavingIntFilter` |  |
| `guidanceDisabledDisplayRgbaValue` | `HavingIntFilter` |  |
| `guidanceUnavailableLedRgbaValue` | `HavingIntFilter` |  |
| `guidanceUnavailableDisplayRgbaValue` | `HavingIntFilter` |  |

## SiteHavingMinInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `organizationId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `guidanceDisabledLedRgbaValue` | `HavingIntFilter` |  |
| `guidanceDisabledDisplayRgbaValue` | `HavingIntFilter` |  |
| `guidanceUnavailableLedRgbaValue` | `HavingIntFilter` |  |
| `guidanceUnavailableDisplayRgbaValue` | `HavingIntFilter` |  |

## SiteHavingStddevPopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `organizationId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `guidanceDisabledLedRgbaValue` | `HavingIntFilter` |  |
| `guidanceDisabledDisplayRgbaValue` | `HavingIntFilter` |  |
| `guidanceUnavailableLedRgbaValue` | `HavingIntFilter` |  |
| `guidanceUnavailableDisplayRgbaValue` | `HavingIntFilter` |  |

## SiteHavingStddevSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `organizationId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `guidanceDisabledLedRgbaValue` | `HavingIntFilter` |  |
| `guidanceDisabledDisplayRgbaValue` | `HavingIntFilter` |  |
| `guidanceUnavailableLedRgbaValue` | `HavingIntFilter` |  |
| `guidanceUnavailableDisplayRgbaValue` | `HavingIntFilter` |  |

## SiteHavingSumInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `organizationId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `guidanceDisabledLedRgbaValue` | `HavingIntFilter` |  |
| `guidanceDisabledDisplayRgbaValue` | `HavingIntFilter` |  |
| `guidanceUnavailableLedRgbaValue` | `HavingIntFilter` |  |
| `guidanceUnavailableDisplayRgbaValue` | `HavingIntFilter` |  |

## SiteHavingVariancePopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `organizationId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `guidanceDisabledLedRgbaValue` | `HavingIntFilter` |  |
| `guidanceDisabledDisplayRgbaValue` | `HavingIntFilter` |  |
| `guidanceUnavailableLedRgbaValue` | `HavingIntFilter` |  |
| `guidanceUnavailableDisplayRgbaValue` | `HavingIntFilter` |  |

## SiteHavingVarianceSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `organizationId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |
| `guidanceDisabledLedRgbaValue` | `HavingIntFilter` |  |
| `guidanceDisabledDisplayRgbaValue` | `HavingIntFilter` |  |
| `guidanceUnavailableLedRgbaValue` | `HavingIntFilter` |  |
| `guidanceUnavailableDisplayRgbaValue` | `HavingIntFilter` |  |

## SiteLevel (object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |
| `id` | `Int!` |  |
| `siteId` | `Int!` |  |
| `position` | `String!` |  |
| `displayName` | `String` |  |
| `createdTimestamp` | `Datetime!` |  |
| `createdUserId` | `Int` |  |
| `createdClientId` | `Int!` |  |
| `lastModifiedTimestamp` | `Datetime!` |  |
| `lastModifiedUserId` | `Int` |  |
| `lastModifiedClientId` | `Int!` |  |
| `isDeleted` | `Boolean!` |  |
| `mapFilePath` | `String` |  |
| `site` | `Site` |  |
| `createdUser` | `User` |  |
| `lastModifiedUser` | `User` |  |
| `parkingZoneDataPoints` | `ParkingZoneDataPointsConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingZoneDataPointsOrderBy!]: [ParkingZoneDataPointsOrderBy!]`, `condition - ParkingZoneDataPointCondition: ParkingZoneDataPointCondition`, `filter - ParkingZoneDataPointFilter: ParkingZoneDataPointFilter` |
| `siteLevelParkingAttributeUsageByHours` | `SiteLevelParkingAttributeUsageByHoursConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [SiteLevelParkingAttributeUsageByHoursOrderBy!]: [SiteLevelParkingAttributeUsageByHoursOrderBy!]`, `condition - SiteLevelParkingAttributeUsageByHourCondition: SiteLevelParkingAttributeUsageByHourCondition`, `filter - SiteLevelParkingAttributeUsageByHourFilter: SiteLevelParkingAttributeUsageByHourFilter` |
| `parkingSpaces` | `ParkingSpacesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingSpacesOrderBy!]: [ParkingSpacesOrderBy!]`, `condition - ParkingSpaceCondition: ParkingSpaceCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - ParkingSpaceFilter: ParkingSpaceFilter` |
| `parkingZones` | `ParkingZonesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingZonesOrderBy!]: [ParkingZonesOrderBy!]`, `condition - ParkingZoneCondition: ParkingZoneCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - ParkingZoneFilter: ParkingZoneFilter` |
| `parkingSpaceDataPoints` | `ParkingSpaceDataPointsConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingSpaceDataPointsOrderBy!]: [ParkingSpaceDataPointsOrderBy!]`, `condition - ParkingSpaceDataPointCondition: ParkingSpaceDataPointCondition`, `filter - ParkingSpaceDataPointFilter: ParkingSpaceDataPointFilter` |
| `latestSiteLevelParkingAttributeParkingUsages` | `LatestSiteLevelParkingAttributeParkingUsagesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [LatestSiteLevelParkingAttributeParkingUsagesOrderBy!]: [LatestSiteLevelParkingAttributeParkingUsagesOrderBy!]`, `condition - LatestSiteLevelParkingAttributeParkingUsageCondition: LatestSiteLevelParkingAttributeParkingUsageCondition`, `filter - LatestSiteLevelParkingAttributeParkingUsageFilter: LatestSiteLevelParkingAttributeParkingUsageFilter` |
| `reportingSiteLevelParkingAttributeUsageByHours` | `ReportingSiteLevelParkingAttributeUsageByHoursConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ReportingSiteLevelParkingAttributeUsageByHoursOrderBy!]: [ReportingSiteLevelParkingAttributeUsageByHoursOrderBy!]`, `condition - ReportingSiteLevelParkingAttributeUsageByHourCondition: ReportingSiteLevelParkingAttributeUsageByHourCondition`, `filter - ReportingSiteLevelParkingAttributeUsageByHourFilter: ReportingSiteLevelParkingAttributeUsageByHourFilter` |
| `mapFilePathSigned` | `String` |  |

## SiteLevelAggregatesFilter (input-object)

Description A filter to be used against aggregates of SiteLevel object types.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `sum` | `SiteLevelSumAggregateFilter` |  |
| `distinctCount` | `SiteLevelDistinctCountAggregateFilter` |  |
| `min` | `SiteLevelMinAggregateFilter` |  |
| `max` | `SiteLevelMaxAggregateFilter` |  |
| `average` | `SiteLevelAverageAggregateFilter` |  |
| `stddevSample` | `SiteLevelStddevSampleAggregateFilter` |  |
| `stddevPopulation` | `SiteLevelStddevPopulationAggregateFilter` |  |
| `varianceSample` | `SiteLevelVarianceSampleAggregateFilter` |  |
| `variancePopulation` | `SiteLevelVariancePopulationAggregateFilter` |  |

## SiteLevelAverageAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |

## SiteLevelCondition (input-object)

Description A condition to be used against SiteLevel object types. All fields are tested for equality and combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `Int` |  |
| `siteId` | `Int` |  |
| `position` | `String` |  |
| `displayName` | `String` |  |
| `createdTimestamp` | `Datetime` |  |
| `createdUserId` | `Int` |  |
| `createdClientId` | `Int` |  |
| `lastModifiedTimestamp` | `Datetime` |  |
| `lastModifiedUserId` | `Int` |  |
| `lastModifiedClientId` | `Int` |  |
| `isDeleted` | `Boolean` |  |
| `mapFilePath` | `String` |  |

## SiteLevelDistinctCountAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `position` | `BigIntFilter` |  |
| `displayName` | `BigIntFilter` |  |
| `createdTimestamp` | `BigIntFilter` |  |
| `createdUserId` | `BigIntFilter` |  |
| `createdClientId` | `BigIntFilter` |  |
| `lastModifiedTimestamp` | `BigIntFilter` |  |
| `lastModifiedUserId` | `BigIntFilter` |  |
| `lastModifiedClientId` | `BigIntFilter` |  |
| `isDeleted` | `BigIntFilter` |  |
| `mapFilePath` | `BigIntFilter` |  |

## SiteLevelFilter (input-object)

Description A filter to be used against SiteLevel object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `siteId` | `IntFilter` |  |
| `position` | `StringFilter` |  |
| `displayName` | `StringFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |
| `isDeleted` | `BooleanFilter` |  |
| `mapFilePath` | `StringFilter` |  |
| `parkingZoneDataPoints` | `SiteLevelToManyParkingZoneDataPointFilter` |  |
| `parkingZoneDataPointsExist` | `Boolean` |  |
| `siteLevelParkingAttributeUsageByHours` | `SiteLevelToManySiteLevelParkingAttributeUsageByHourFilter` |  |
| `siteLevelParkingAttributeUsageByHoursExist` | `Boolean` |  |
| `parkingSpaces` | `SiteLevelToManyParkingSpaceFilter` |  |
| `parkingSpacesExist` | `Boolean` |  |
| `parkingZones` | `SiteLevelToManyParkingZoneFilter` |  |
| `parkingZonesExist` | `Boolean` |  |
| `parkingSpaceDataPoints` | `SiteLevelToManyParkingSpaceDataPointFilter` |  |
| `parkingSpaceDataPointsExist` | `Boolean` |  |
| `automationRuleConditionSiteLevelsExist` | `Boolean` |  |
| `latestSiteLevelParkingAttributeParkingUsages` | `SiteLevelToManyLatestSiteLevelParkingAttributeParkingUsageFilter` |  |
| `latestSiteLevelParkingAttributeParkingUsagesExist` | `Boolean` |  |
| `latestSiteLevelParkingUsagesExist` | `Boolean` |  |
| `reportingSiteLevelParkingAttributeUsageByHours` | `SiteLevelToManyReportingSiteLevelParkingAttributeUsageByHourFilter` |  |
| `reportingSiteLevelParkingAttributeUsageByHoursExist` | `Boolean` |  |
| `createdUserExists` | `Boolean` |  |
| `lastModifiedUserExists` | `Boolean` |  |

## SiteLevelHavingAverageInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## SiteLevelHavingDistinctCountInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## SiteLevelHavingInput (input-object)

Description Conditions for SiteLevel aggregates.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `AND` | `[SiteLevelHavingInput!]` |  |
| `OR` | `[SiteLevelHavingInput!]` |  |
| `sum` | `SiteLevelHavingSumInput` |  |
| `distinctCount` | `SiteLevelHavingDistinctCountInput` |  |
| `min` | `SiteLevelHavingMinInput` |  |
| `max` | `SiteLevelHavingMaxInput` |  |
| `average` | `SiteLevelHavingAverageInput` |  |
| `stddevSample` | `SiteLevelHavingStddevSampleInput` |  |
| `stddevPopulation` | `SiteLevelHavingStddevPopulationInput` |  |
| `varianceSample` | `SiteLevelHavingVarianceSampleInput` |  |
| `variancePopulation` | `SiteLevelHavingVariancePopulationInput` |  |

## SiteLevelHavingMaxInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## SiteLevelHavingMinInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## SiteLevelHavingStddevPopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## SiteLevelHavingStddevSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## SiteLevelHavingSumInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## SiteLevelHavingVariancePopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## SiteLevelHavingVarianceSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## SiteLevelMaxAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `siteId` | `IntFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |

## SiteLevelMinAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `siteId` | `IntFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |

## SiteLevelParkingAttributeUsageByHour (object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |
| `siteId` | `Int!` |  |
| `siteLevelId` | `Int!` |  |
| `parkingAttributeId` | `Int!` |  |
| `hour` | `Datetime!` |  |
| `availableCount` | `BigFloat!` |  |
| `noDataCount` | `BigFloat!` |  |
| `occupiedCount` | `BigFloat!` |  |
| `totalCount` | `BigFloat!` |  |
| `finalizedAt` | `Datetime` |  |
| `site` | `Site` |  |
| `siteLevel` | `SiteLevel` |  |
| `parkingAttribute` | `ParkingAttribute` |  |

## SiteLevelParkingAttributeUsageByHourAggregatesFilter (input-object)

Description A filter to be used against aggregates of SiteLevelParkingAttributeUsageByHour object types.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `sum` | `SiteLevelParkingAttributeUsageByHourSumAggregateFilter` |  |
| `distinctCount` | `SiteLevelParkingAttributeUsageByHourDistinctCountAggregateFilter` |  |
| `min` | `SiteLevelParkingAttributeUsageByHourMinAggregateFilter` |  |
| `max` | `SiteLevelParkingAttributeUsageByHourMaxAggregateFilter` |  |
| `average` | `SiteLevelParkingAttributeUsageByHourAverageAggregateFilter` |  |
| `stddevSample` | `SiteLevelParkingAttributeUsageByHourStddevSampleAggregateFilter` |  |
| `stddevPopulation` | `SiteLevelParkingAttributeUsageByHourStddevPopulationAggregateFilter` |  |
| `varianceSample` | `SiteLevelParkingAttributeUsageByHourVarianceSampleAggregateFilter` |  |
| `variancePopulation` | `SiteLevelParkingAttributeUsageByHourVariancePopulationAggregateFilter` |  |

## SiteLevelParkingAttributeUsageByHourAverageAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `BigFloatFilter` |  |
| `siteLevelId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |

## SiteLevelParkingAttributeUsageByHourCondition (input-object)

Description A condition to be used against SiteLevelParkingAttributeUsageByHour object types. All fields are tested for equality and combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `Int` |  |
| `siteLevelId` | `Int` |  |
| `parkingAttributeId` | `Int` |  |
| `hour` | `Datetime` |  |
| `availableCount` | `BigFloat` |  |
| `noDataCount` | `BigFloat` |  |
| `occupiedCount` | `BigFloat` |  |
| `totalCount` | `BigFloat` |  |
| `finalizedAt` | `Datetime` |  |

## SiteLevelParkingAttributeUsageByHourDistinctCountAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `BigIntFilter` |  |
| `siteLevelId` | `BigIntFilter` |  |
| `parkingAttributeId` | `BigIntFilter` |  |
| `hour` | `BigIntFilter` |  |
| `availableCount` | `BigIntFilter` |  |
| `noDataCount` | `BigIntFilter` |  |
| `occupiedCount` | `BigIntFilter` |  |
| `totalCount` | `BigIntFilter` |  |
| `finalizedAt` | `BigIntFilter` |  |

## SiteLevelParkingAttributeUsageByHourFilter (input-object)

Description A filter to be used against SiteLevelParkingAttributeUsageByHour object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `IntFilter` |  |
| `siteLevelId` | `IntFilter` |  |
| `parkingAttributeId` | `IntFilter` |  |
| `hour` | `DatetimeFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `finalizedAt` | `DatetimeFilter` |  |

## SiteLevelParkingAttributeUsageByHourHavingAverageInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `hour` | `HavingDatetimeFilter` |  |
| `finalizedAt` | `HavingDatetimeFilter` |  |

## SiteLevelParkingAttributeUsageByHourHavingDistinctCountInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `hour` | `HavingDatetimeFilter` |  |
| `finalizedAt` | `HavingDatetimeFilter` |  |

## SiteLevelParkingAttributeUsageByHourHavingInput (input-object)

Description Conditions for SiteLevelParkingAttributeUsageByHour aggregates.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `AND` | `[SiteLevelParkingAttributeUsageByHourHavingInput!]` |  |
| `OR` | `[SiteLevelParkingAttributeUsageByHourHavingInput!]` |  |
| `sum` | `SiteLevelParkingAttributeUsageByHourHavingSumInput` |  |
| `distinctCount` | `SiteLevelParkingAttributeUsageByHourHavingDistinctCountInput` |  |
| `min` | `SiteLevelParkingAttributeUsageByHourHavingMinInput` |  |
| `max` | `SiteLevelParkingAttributeUsageByHourHavingMaxInput` |  |
| `average` | `SiteLevelParkingAttributeUsageByHourHavingAverageInput` |  |
| `stddevSample` | `SiteLevelParkingAttributeUsageByHourHavingStddevSampleInput` |  |
| `stddevPopulation` | `SiteLevelParkingAttributeUsageByHourHavingStddevPopulationInput` |  |
| `varianceSample` | `SiteLevelParkingAttributeUsageByHourHavingVarianceSampleInput` |  |
| `variancePopulation` | `SiteLevelParkingAttributeUsageByHourHavingVariancePopulationInput` |  |

## SiteLevelParkingAttributeUsageByHourHavingMaxInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `hour` | `HavingDatetimeFilter` |  |
| `finalizedAt` | `HavingDatetimeFilter` |  |

## SiteLevelParkingAttributeUsageByHourHavingMinInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `hour` | `HavingDatetimeFilter` |  |
| `finalizedAt` | `HavingDatetimeFilter` |  |

## SiteLevelParkingAttributeUsageByHourHavingStddevPopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `hour` | `HavingDatetimeFilter` |  |
| `finalizedAt` | `HavingDatetimeFilter` |  |

## SiteLevelParkingAttributeUsageByHourHavingStddevSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `hour` | `HavingDatetimeFilter` |  |
| `finalizedAt` | `HavingDatetimeFilter` |  |

## SiteLevelParkingAttributeUsageByHourHavingSumInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `hour` | `HavingDatetimeFilter` |  |
| `finalizedAt` | `HavingDatetimeFilter` |  |

## SiteLevelParkingAttributeUsageByHourHavingVariancePopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `hour` | `HavingDatetimeFilter` |  |
| `finalizedAt` | `HavingDatetimeFilter` |  |

## SiteLevelParkingAttributeUsageByHourHavingVarianceSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `HavingIntFilter` |  |
| `siteLevelId` | `HavingIntFilter` |  |
| `parkingAttributeId` | `HavingIntFilter` |  |
| `hour` | `HavingDatetimeFilter` |  |
| `finalizedAt` | `HavingDatetimeFilter` |  |

## SiteLevelParkingAttributeUsageByHourMaxAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `IntFilter` |  |
| `siteLevelId` | `IntFilter` |  |
| `parkingAttributeId` | `IntFilter` |  |
| `hour` | `DatetimeFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `finalizedAt` | `DatetimeFilter` |  |

## SiteLevelParkingAttributeUsageByHourMinAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `IntFilter` |  |
| `siteLevelId` | `IntFilter` |  |
| `parkingAttributeId` | `IntFilter` |  |
| `hour` | `DatetimeFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |
| `finalizedAt` | `DatetimeFilter` |  |

## SiteLevelParkingAttributeUsageByHourStddevPopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `BigFloatFilter` |  |
| `siteLevelId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |

## SiteLevelParkingAttributeUsageByHourStddevSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `BigFloatFilter` |  |
| `siteLevelId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |

## SiteLevelParkingAttributeUsageByHourSumAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `BigIntFilter` |  |
| `siteLevelId` | `BigIntFilter` |  |
| `parkingAttributeId` | `BigIntFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |

## SiteLevelParkingAttributeUsageByHourVariancePopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `BigFloatFilter` |  |
| `siteLevelId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |

## SiteLevelParkingAttributeUsageByHourVarianceSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `siteId` | `BigFloatFilter` |  |
| `siteLevelId` | `BigFloatFilter` |  |
| `parkingAttributeId` | `BigFloatFilter` |  |
| `availableCount` | `BigFloatFilter` |  |
| `noDataCount` | `BigFloatFilter` |  |
| `occupiedCount` | `BigFloatFilter` |  |
| `totalCount` | `BigFloatFilter` |  |

## SiteLevelParkingAttributeUsageByHoursConnection (object)

Description A connection to a list of SiteLevelParkingAttributeUsageByHour values.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodes` | `[SiteLevelParkingAttributeUsageByHour!]!` |  |
| `totalCount` | `Int!` |  |

## SiteLevelParkingAttributeUsageByHoursOrderBy (enum)

Description Methods to use when ordering SiteLevelParkingAttributeUsageByHour .

## SiteLevelStddevPopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |

## SiteLevelStddevSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |

## SiteLevelSumAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `createdUserId` | `BigIntFilter` |  |
| `createdClientId` | `BigIntFilter` |  |
| `lastModifiedUserId` | `BigIntFilter` |  |
| `lastModifiedClientId` | `BigIntFilter` |  |

## SiteLevelToManyLatestSiteLevelParkingAttributeParkingUsageFilter (input-object)

Description A filter to be used against many LatestSiteLevelParkingAttributeParkingUsage object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `LatestSiteLevelParkingAttributeParkingUsageAggregatesFilter` |  |

## SiteLevelToManyParkingSpaceDataPointFilter (input-object)

Description A filter to be used against many ParkingSpaceDataPoint object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingSpaceDataPointAggregatesFilter` |  |

## SiteLevelToManyParkingSpaceFilter (input-object)

Description A filter to be used against many ParkingSpace object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingSpaceAggregatesFilter` |  |

## SiteLevelToManyParkingZoneDataPointFilter (input-object)

Description A filter to be used against many ParkingZoneDataPoint object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingZoneDataPointAggregatesFilter` |  |

## SiteLevelToManyParkingZoneFilter (input-object)

Description A filter to be used against many ParkingZone object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingZoneAggregatesFilter` |  |

## SiteLevelToManyReportingSiteLevelParkingAttributeUsageByHourFilter (input-object)

Description A filter to be used against many ReportingSiteLevelParkingAttributeUsageByHour object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ReportingSiteLevelParkingAttributeUsageByHourAggregatesFilter` |  |

## SiteLevelToManySiteLevelParkingAttributeUsageByHourFilter (input-object)

Description A filter to be used against many SiteLevelParkingAttributeUsageByHour object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `SiteLevelParkingAttributeUsageByHourAggregatesFilter` |  |

## SiteLevelVariancePopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |

## SiteLevelVarianceSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |

## SiteLevelsConnection (object)

Description A connection to a list of SiteLevel values.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodes` | `[SiteLevel!]!` |  |
| `totalCount` | `Int!` |  |

## SiteLevelsOrderBy (enum)

Description Methods to use when ordering SiteLevel .

## SiteMaxAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `organizationId` | `IntFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |
| `guidanceDisabledLedRgbaValue` | `IntFilter` |  |
| `guidanceDisabledDisplayRgbaValue` | `IntFilter` |  |
| `guidanceUnavailableLedRgbaValue` | `IntFilter` |  |
| `guidanceUnavailableDisplayRgbaValue` | `IntFilter` |  |

## SiteMinAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `organizationId` | `IntFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |
| `guidanceDisabledLedRgbaValue` | `IntFilter` |  |
| `guidanceDisabledDisplayRgbaValue` | `IntFilter` |  |
| `guidanceUnavailableLedRgbaValue` | `IntFilter` |  |
| `guidanceUnavailableDisplayRgbaValue` | `IntFilter` |  |

## SiteStddevPopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `organizationId` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |
| `guidanceDisabledLedRgbaValue` | `BigFloatFilter` |  |
| `guidanceDisabledDisplayRgbaValue` | `BigFloatFilter` |  |
| `guidanceUnavailableLedRgbaValue` | `BigFloatFilter` |  |
| `guidanceUnavailableDisplayRgbaValue` | `BigFloatFilter` |  |

## SiteStddevSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `organizationId` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |
| `guidanceDisabledLedRgbaValue` | `BigFloatFilter` |  |
| `guidanceDisabledDisplayRgbaValue` | `BigFloatFilter` |  |
| `guidanceUnavailableLedRgbaValue` | `BigFloatFilter` |  |
| `guidanceUnavailableDisplayRgbaValue` | `BigFloatFilter` |  |

## SiteSumAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `organizationId` | `BigIntFilter` |  |
| `createdUserId` | `BigIntFilter` |  |
| `createdClientId` | `BigIntFilter` |  |
| `lastModifiedUserId` | `BigIntFilter` |  |
| `lastModifiedClientId` | `BigIntFilter` |  |
| `guidanceDisabledLedRgbaValue` | `BigIntFilter` |  |
| `guidanceDisabledDisplayRgbaValue` | `BigIntFilter` |  |
| `guidanceUnavailableLedRgbaValue` | `BigIntFilter` |  |
| `guidanceUnavailableDisplayRgbaValue` | `BigIntFilter` |  |

## SiteToManyLatestSiteLevelParkingAttributeParkingUsageFilter (input-object)

Description A filter to be used against many LatestSiteLevelParkingAttributeParkingUsage object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `LatestSiteLevelParkingAttributeParkingUsageAggregatesFilter` |  |

## SiteToManyLatestSiteParkingAttributeParkingUsageFilter (input-object)

Description A filter to be used against many LatestSiteParkingAttributeParkingUsage object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `LatestSiteParkingAttributeParkingUsageAggregatesFilter` |  |

## SiteToManyLatestSiteParkingUsageFilter (input-object)

Description A filter to be used against many LatestSiteParkingUsage object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `LatestSiteParkingUsageAggregatesFilter` |  |

## SiteToManyParkingAttributeFilter (input-object)

Description A filter to be used against many ParkingAttribute object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingAttributeAggregatesFilter` |  |

## SiteToManyParkingSpaceDataPointFilter (input-object)

Description A filter to be used against many ParkingSpaceDataPoint object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingSpaceDataPointAggregatesFilter` |  |

## SiteToManyParkingSpaceFilter (input-object)

Description A filter to be used against many ParkingSpace object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingSpaceAggregatesFilter` |  |

## SiteToManyParkingSpaceVehicleSessionFilter (input-object)

Description A filter to be used against many ParkingSpaceVehicleSession object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingSpaceVehicleSessionAggregatesFilter` |  |

## SiteToManyParkingSpaceVehicleSessionVehicleRecognitionFilter (input-object)

Description A filter to be used against many ParkingSpaceVehicleSessionVehicleRecognition object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingSpaceVehicleSessionVehicleRecognitionAggregatesFilter` |  |

## SiteToManyParkingZoneCounterFilter (input-object)

Description A filter to be used against many ParkingZoneCounter object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingZoneCounterAggregatesFilter` |  |

## SiteToManyParkingZoneDataPointFilter (input-object)

Description A filter to be used against many ParkingZoneDataPoint object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingZoneDataPointAggregatesFilter` |  |

## SiteToManyParkingZoneFilter (input-object)

Description A filter to be used against many ParkingZone object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingZoneAggregatesFilter` |  |

## SiteToManyReportingSiteLevelParkingAttributeUsageByHourFilter (input-object)

Description A filter to be used against many ReportingSiteLevelParkingAttributeUsageByHour object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ReportingSiteLevelParkingAttributeUsageByHourAggregatesFilter` |  |

## SiteToManySensorFilter (input-object)

Description A filter to be used against many Sensor object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `SensorAggregatesFilter` |  |

## SiteToManySiteLevelFilter (input-object)

Description A filter to be used against many SiteLevel object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `SiteLevelAggregatesFilter` |  |

## SiteToManySiteLevelParkingAttributeUsageByHourFilter (input-object)

Description A filter to be used against many SiteLevelParkingAttributeUsageByHour object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `SiteLevelParkingAttributeUsageByHourAggregatesFilter` |  |

## SiteToManyVehicleRecognitionFilter (input-object)

Description A filter to be used against many VehicleRecognition object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `VehicleRecognitionAggregatesFilter` |  |

## SiteToManyVehicleRecognitionPlateFilter (input-object)

Description A filter to be used against many VehicleRecognitionPlate object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `VehicleRecognitionPlateAggregatesFilter` |  |

## SiteVariancePopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `organizationId` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |
| `guidanceDisabledLedRgbaValue` | `BigFloatFilter` |  |
| `guidanceDisabledDisplayRgbaValue` | `BigFloatFilter` |  |
| `guidanceUnavailableLedRgbaValue` | `BigFloatFilter` |  |
| `guidanceUnavailableDisplayRgbaValue` | `BigFloatFilter` |  |

## SiteVarianceSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `organizationId` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |
| `guidanceDisabledLedRgbaValue` | `BigFloatFilter` |  |
| `guidanceDisabledDisplayRgbaValue` | `BigFloatFilter` |  |
| `guidanceUnavailableLedRgbaValue` | `BigFloatFilter` |  |
| `guidanceUnavailableDisplayRgbaValue` | `BigFloatFilter` |  |

## SitesConnection (object)

Description A connection to a list of Site values.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodes` | `[Site!]!` |  |
| `totalCount` | `Int!` |  |

## SitesOrderBy (enum)

Description Methods to use when ordering Site .

## String (scalar)

Description The String scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text.

## StringFilter (input-object)

Description A filter to be used against String fields. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `isNull` | `Boolean` |  |
| `equalTo` | `String` |  |
| `notEqualTo` | `String` |  |
| `distinctFrom` | `String` |  |
| `notDistinctFrom` | `String` |  |
| `in` | `[String!]` |  |
| `notIn` | `[String!]` |  |
| `lessThan` | `String` |  |
| `lessThanOrEqualTo` | `String` |  |
| `greaterThan` | `String` |  |
| `greaterThanOrEqualTo` | `String` |  |
| `includes` | `String` |  |
| `notIncludes` | `String` |  |
| `includesInsensitive` | `String` |  |
| `notIncludesInsensitive` | `String` |  |
| `startsWith` | `String` |  |
| `notStartsWith` | `String` |  |
| `startsWithInsensitive` | `String` |  |
| `notStartsWithInsensitive` | `String` |  |
| `endsWith` | `String` |  |
| `notEndsWith` | `String` |  |
| `endsWithInsensitive` | `String` |  |
| `notEndsWithInsensitive` | `String` |  |
| `like` | `String` |  |
| `notLike` | `String` |  |
| `likeInsensitive` | `String` |  |
| `notLikeInsensitive` | `String` |  |
| `equalToInsensitive` | `String` |  |
| `notEqualToInsensitive` | `String` |  |
| `distinctFromInsensitive` | `String` |  |
| `notDistinctFromInsensitive` | `String` |  |
| `inInsensitive` | `[String!]` |  |
| `notInInsensitive` | `[String!]` |  |
| `lessThanInsensitive` | `String` |  |
| `lessThanOrEqualToInsensitive` | `String` |  |
| `greaterThanInsensitive` | `String` |  |
| `greaterThanOrEqualToInsensitive` | `String` |  |

## UUID (scalar)

Description A universally unique identifier as defined by RFC 4122 .

## User (object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |
| `id` | `Int!` |  |
| `email` | `String!` |  |
| `givenName` | `String!` |  |
| `familyName` | `String` |  |
| `createdTimestamp` | `Datetime!` |  |
| `createdUserId` | `Int` |  |
| `createdClientId` | `Int!` |  |
| `lastModifiedTimestamp` | `Datetime!` |  |
| `lastModifiedUserId` | `Int` |  |
| `lastModifiedClientId` | `Int!` |  |
| `subjectId` | `UUID!` |  |
| `isDeleted` | `Boolean!` |  |
| `createdUser` | `User` |  |
| `lastModifiedUser` | `User` |  |
| `usersByCreatedUserId` | `UsersConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [UsersOrderBy!]: [UsersOrderBy!]`, `condition - UserCondition: UserCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - UserFilter: UserFilter` |
| `usersByLastModifiedUserId` | `UsersConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [UsersOrderBy!]: [UsersOrderBy!]`, `condition - UserCondition: UserCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - UserFilter: UserFilter` |
| `organizationsByCreatedUserId` | `OrganizationsConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [OrganizationsOrderBy!]: [OrganizationsOrderBy!]`, `condition - OrganizationCondition: OrganizationCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - OrganizationFilter: OrganizationFilter` |
| `organizationsByLastModifiedUserId` | `OrganizationsConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [OrganizationsOrderBy!]: [OrganizationsOrderBy!]`, `condition - OrganizationCondition: OrganizationCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - OrganizationFilter: OrganizationFilter` |
| `organizationUsersByUserId` | `OrganizationUsersConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [OrganizationUsersOrderBy!]: [OrganizationUsersOrderBy!]`, `condition - OrganizationUserCondition: OrganizationUserCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - OrganizationUserFilter: OrganizationUserFilter` |
| `organizationUsersByCreatedUserId` | `OrganizationUsersConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [OrganizationUsersOrderBy!]: [OrganizationUsersOrderBy!]`, `condition - OrganizationUserCondition: OrganizationUserCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - OrganizationUserFilter: OrganizationUserFilter` |
| `organizationUsersByLastModifiedUserId` | `OrganizationUsersConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [OrganizationUsersOrderBy!]: [OrganizationUsersOrderBy!]`, `condition - OrganizationUserCondition: OrganizationUserCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - OrganizationUserFilter: OrganizationUserFilter` |
| `parkingAttributesByCreatedUserId` | `ParkingAttributesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingAttributesOrderBy!]: [ParkingAttributesOrderBy!]`, `condition - ParkingAttributeCondition: ParkingAttributeCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - ParkingAttributeFilter: ParkingAttributeFilter` |
| `parkingAttributesByLastModifiedUserId` | `ParkingAttributesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingAttributesOrderBy!]: [ParkingAttributesOrderBy!]`, `condition - ParkingAttributeCondition: ParkingAttributeCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - ParkingAttributeFilter: ParkingAttributeFilter` |
| `parkingSpacesByCreatedUserId` | `ParkingSpacesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingSpacesOrderBy!]: [ParkingSpacesOrderBy!]`, `condition - ParkingSpaceCondition: ParkingSpaceCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - ParkingSpaceFilter: ParkingSpaceFilter` |
| `parkingSpacesByLastModifiedUserId` | `ParkingSpacesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingSpacesOrderBy!]: [ParkingSpacesOrderBy!]`, `condition - ParkingSpaceCondition: ParkingSpaceCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - ParkingSpaceFilter: ParkingSpaceFilter` |
| `parkingZonesByCreatedUserId` | `ParkingZonesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingZonesOrderBy!]: [ParkingZonesOrderBy!]`, `condition - ParkingZoneCondition: ParkingZoneCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - ParkingZoneFilter: ParkingZoneFilter` |
| `parkingZonesByLastModifiedUserId` | `ParkingZonesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingZonesOrderBy!]: [ParkingZonesOrderBy!]`, `condition - ParkingZoneCondition: ParkingZoneCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - ParkingZoneFilter: ParkingZoneFilter` |
| `sitesByCreatedUserId` | `SitesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [SitesOrderBy!]: [SitesOrderBy!]`, `condition - SiteCondition: SiteCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - SiteFilter: SiteFilter` |
| `sitesByLastModifiedUserId` | `SitesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [SitesOrderBy!]: [SitesOrderBy!]`, `condition - SiteCondition: SiteCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - SiteFilter: SiteFilter` |
| `siteLevelsByCreatedUserId` | `SiteLevelsConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [SiteLevelsOrderBy!]: [SiteLevelsOrderBy!]`, `condition - SiteLevelCondition: SiteLevelCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - SiteLevelFilter: SiteLevelFilter` |
| `siteLevelsByLastModifiedUserId` | `SiteLevelsConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [SiteLevelsOrderBy!]: [SiteLevelsOrderBy!]`, `condition - SiteLevelCondition: SiteLevelCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - SiteLevelFilter: SiteLevelFilter` |
| `organizationRolesByCreatedUserId` | `OrganizationRolesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [OrganizationRolesOrderBy!]: [OrganizationRolesOrderBy!]`, `condition - OrganizationRoleCondition: OrganizationRoleCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - OrganizationRoleFilter: OrganizationRoleFilter` |
| `organizationRolesByLastModifiedUserId` | `OrganizationRolesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [OrganizationRolesOrderBy!]: [OrganizationRolesOrderBy!]`, `condition - OrganizationRoleCondition: OrganizationRoleCondition`, `includeDeleted - IncludeDeletedOption: IncludeDeletedOption`, `filter - OrganizationRoleFilter: OrganizationRoleFilter` |

## UserAggregatesFilter (input-object)

Description A filter to be used against aggregates of User object types.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `sum` | `UserSumAggregateFilter` |  |
| `distinctCount` | `UserDistinctCountAggregateFilter` |  |
| `min` | `UserMinAggregateFilter` |  |
| `max` | `UserMaxAggregateFilter` |  |
| `average` | `UserAverageAggregateFilter` |  |
| `stddevSample` | `UserStddevSampleAggregateFilter` |  |
| `stddevPopulation` | `UserStddevPopulationAggregateFilter` |  |
| `varianceSample` | `UserVarianceSampleAggregateFilter` |  |
| `variancePopulation` | `UserVariancePopulationAggregateFilter` |  |

## UserAverageAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |

## UserCondition (input-object)

Description A condition to be used against User object types. All fields are tested for equality and combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `Int` |  |
| `email` | `String` |  |
| `givenName` | `String` |  |
| `familyName` | `String` |  |
| `createdTimestamp` | `Datetime` |  |
| `createdUserId` | `Int` |  |
| `createdClientId` | `Int` |  |
| `lastModifiedTimestamp` | `Datetime` |  |
| `lastModifiedUserId` | `Int` |  |
| `lastModifiedClientId` | `Int` |  |
| `subjectId` | `UUID` |  |
| `isDeleted` | `Boolean` |  |

## UserDistinctCountAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `email` | `BigIntFilter` |  |
| `givenName` | `BigIntFilter` |  |
| `familyName` | `BigIntFilter` |  |
| `createdTimestamp` | `BigIntFilter` |  |
| `createdUserId` | `BigIntFilter` |  |
| `createdClientId` | `BigIntFilter` |  |
| `lastModifiedTimestamp` | `BigIntFilter` |  |
| `lastModifiedUserId` | `BigIntFilter` |  |
| `lastModifiedClientId` | `BigIntFilter` |  |
| `subjectId` | `BigIntFilter` |  |
| `isDeleted` | `BigIntFilter` |  |

## UserFilter (input-object)

Description A filter to be used against User object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `email` | `StringFilter` |  |
| `givenName` | `StringFilter` |  |
| `familyName` | `StringFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |
| `isDeleted` | `BooleanFilter` |  |
| `usersByCreatedUserId` | `UserToManyUserFilter` |  |
| `usersByCreatedUserIdExist` | `Boolean` |  |
| `usersByLastModifiedUserId` | `UserToManyUserFilter` |  |
| `usersByLastModifiedUserIdExist` | `Boolean` |  |
| `organizationsByCreatedUserId` | `UserToManyOrganizationFilter` |  |
| `organizationsByCreatedUserIdExist` | `Boolean` |  |
| `organizationsByLastModifiedUserId` | `UserToManyOrganizationFilter` |  |
| `organizationsByLastModifiedUserIdExist` | `Boolean` |  |
| `organizationUsersByUserId` | `UserToManyOrganizationUserFilter` |  |
| `organizationUsersByUserIdExist` | `Boolean` |  |
| `organizationUsersByCreatedUserId` | `UserToManyOrganizationUserFilter` |  |
| `organizationUsersByCreatedUserIdExist` | `Boolean` |  |
| `organizationUsersByLastModifiedUserId` | `UserToManyOrganizationUserFilter` |  |
| `organizationUsersByLastModifiedUserIdExist` | `Boolean` |  |
| `systemUsersByUserIdExist` | `Boolean` |  |
| `systemUsersByCreatedUserIdExist` | `Boolean` |  |
| `systemUsersByLastModifiedUserIdExist` | `Boolean` |  |
| `organizationPublicApiQueriesByCreatedUserIdExist` | `Boolean` |  |
| `organizationPublicApiQueriesByLastModifiedUserIdExist` | `Boolean` |  |
| `apiTokensByUserIdExist` | `Boolean` |  |
| `parkingAttributesByCreatedUserId` | `UserToManyParkingAttributeFilter` |  |
| `parkingAttributesByCreatedUserIdExist` | `Boolean` |  |
| `parkingAttributesByLastModifiedUserId` | `UserToManyParkingAttributeFilter` |  |
| `parkingAttributesByLastModifiedUserIdExist` | `Boolean` |  |
| `parkingSpacesByCreatedUserId` | `UserToManyParkingSpaceFilter` |  |
| `parkingSpacesByCreatedUserIdExist` | `Boolean` |  |
| `parkingSpacesByLastModifiedUserId` | `UserToManyParkingSpaceFilter` |  |
| `parkingSpacesByLastModifiedUserIdExist` | `Boolean` |  |
| `parkingZonesByCreatedUserId` | `UserToManyParkingZoneFilter` |  |
| `parkingZonesByCreatedUserIdExist` | `Boolean` |  |
| `parkingZonesByLastModifiedUserId` | `UserToManyParkingZoneFilter` |  |
| `parkingZonesByLastModifiedUserIdExist` | `Boolean` |  |
| `sitesByCreatedUserId` | `UserToManySiteFilter` |  |
| `sitesByCreatedUserIdExist` | `Boolean` |  |
| `sitesByLastModifiedUserId` | `UserToManySiteFilter` |  |
| `sitesByLastModifiedUserIdExist` | `Boolean` |  |
| `siteLevelsByCreatedUserId` | `UserToManySiteLevelFilter` |  |
| `siteLevelsByCreatedUserIdExist` | `Boolean` |  |
| `siteLevelsByLastModifiedUserId` | `UserToManySiteLevelFilter` |  |
| `siteLevelsByLastModifiedUserIdExist` | `Boolean` |  |
| `organizationRolesByCreatedUserId` | `UserToManyOrganizationRoleFilter` |  |
| `organizationRolesByCreatedUserIdExist` | `Boolean` |  |
| `organizationRolesByLastModifiedUserId` | `UserToManyOrganizationRoleFilter` |  |
| `organizationRolesByLastModifiedUserIdExist` | `Boolean` |  |
| `notificationsByUserIdExist` | `Boolean` |  |
| `userPreferenceByUserIdExists` | `Boolean` |  |
| `notificationReadsByUserIdExist` | `Boolean` |  |
| `validationParkingSpaceResponseOverridesByUserIdExist` | `Boolean` |  |
| `parkingSpaceVehicleSessionPlatesByUserIdExist` | `Boolean` |  |
| `vehicleSessionPlatesByUserIdExist` | `Boolean` |  |
| `pageViewsByUserIdExist` | `Boolean` |  |
| `userSessionsByUserIdExist` | `Boolean` |  |
| `createdUserExists` | `Boolean` |  |
| `lastModifiedUserExists` | `Boolean` |  |

## UserHavingAverageInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## UserHavingDistinctCountInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## UserHavingInput (input-object)

Description Conditions for User aggregates.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `AND` | `[UserHavingInput!]` |  |
| `OR` | `[UserHavingInput!]` |  |
| `sum` | `UserHavingSumInput` |  |
| `distinctCount` | `UserHavingDistinctCountInput` |  |
| `min` | `UserHavingMinInput` |  |
| `max` | `UserHavingMaxInput` |  |
| `average` | `UserHavingAverageInput` |  |
| `stddevSample` | `UserHavingStddevSampleInput` |  |
| `stddevPopulation` | `UserHavingStddevPopulationInput` |  |
| `varianceSample` | `UserHavingVarianceSampleInput` |  |
| `variancePopulation` | `UserHavingVariancePopulationInput` |  |

## UserHavingMaxInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## UserHavingMinInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## UserHavingStddevPopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## UserHavingStddevSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## UserHavingSumInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## UserHavingVariancePopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## UserHavingVarianceSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingIntFilter` |  |
| `createdTimestamp` | `HavingDatetimeFilter` |  |
| `createdUserId` | `HavingIntFilter` |  |
| `createdClientId` | `HavingIntFilter` |  |
| `lastModifiedTimestamp` | `HavingDatetimeFilter` |  |
| `lastModifiedUserId` | `HavingIntFilter` |  |
| `lastModifiedClientId` | `HavingIntFilter` |  |

## UserMaxAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |

## UserMinAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `IntFilter` |  |
| `createdTimestamp` | `DatetimeFilter` |  |
| `createdUserId` | `IntFilter` |  |
| `createdClientId` | `IntFilter` |  |
| `lastModifiedTimestamp` | `DatetimeFilter` |  |
| `lastModifiedUserId` | `IntFilter` |  |
| `lastModifiedClientId` | `IntFilter` |  |

## UserStddevPopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |

## UserStddevSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |

## UserSumAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `createdUserId` | `BigIntFilter` |  |
| `createdClientId` | `BigIntFilter` |  |
| `lastModifiedUserId` | `BigIntFilter` |  |
| `lastModifiedClientId` | `BigIntFilter` |  |

## UserToManyOrganizationFilter (input-object)

Description A filter to be used against many Organization object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `OrganizationAggregatesFilter` |  |

## UserToManyOrganizationRoleFilter (input-object)

Description A filter to be used against many OrganizationRole object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `OrganizationRoleAggregatesFilter` |  |

## UserToManyOrganizationUserFilter (input-object)

Description A filter to be used against many OrganizationUser object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `OrganizationUserAggregatesFilter` |  |

## UserToManyParkingAttributeFilter (input-object)

Description A filter to be used against many ParkingAttribute object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingAttributeAggregatesFilter` |  |

## UserToManyParkingSpaceFilter (input-object)

Description A filter to be used against many ParkingSpace object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingSpaceAggregatesFilter` |  |

## UserToManyParkingZoneFilter (input-object)

Description A filter to be used against many ParkingZone object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingZoneAggregatesFilter` |  |

## UserToManySiteFilter (input-object)

Description A filter to be used against many Site object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `SiteAggregatesFilter` |  |

## UserToManySiteLevelFilter (input-object)

Description A filter to be used against many SiteLevel object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `SiteLevelAggregatesFilter` |  |

## UserToManyUserFilter (input-object)

Description A filter to be used against many User object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `UserAggregatesFilter` |  |

## UserVariancePopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |

## UserVarianceSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `createdUserId` | `BigFloatFilter` |  |
| `createdClientId` | `BigFloatFilter` |  |
| `lastModifiedUserId` | `BigFloatFilter` |  |
| `lastModifiedClientId` | `BigFloatFilter` |  |

## UsersConnection (object)

Description A connection to a list of User values.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodes` | `[User!]!` |  |
| `totalCount` | `Int!` |  |

## UsersOrderBy (enum)

Description Methods to use when ordering User .

## VehicleRecognition (object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |
| `id` | `BigInt!` |  |
| `remoteId` | `Int!` |  |
| `timestamp` | `Datetime!` |  |
| `dscore` | `BigFloat!` |  |
| `location` | `String!` |  |
| `siteId` | `Int!` |  |
| `cameraId` | `Int` |  |
| `remoteImagePath` | `String` |  |
| `imageUrl` | `String` |  |
| `remoteUpdatedAt` | `Datetime!` |  |
| `remoteMakeModelImagePath` | `String` |  |
| `makeModelImageUrl` | `String` |  |
| `plateBoxXMin` | `Int` |  |
| `plateBoxXMax` | `Int` |  |
| `plateBoxYMin` | `Int` |  |
| `plateBoxYMax` | `Int` |  |
| `imageWidth` | `Int` |  |
| `imageHeight` | `Int` |  |
| `compositeImageUrl` | `String` |  |
| `remoteChangeSeqId` | `BigInt` |  |
| `updatedAt` | `Datetime!` |  |
| `site` | `Site` |  |
| `vehicleRecognitionPlates` | `VehicleRecognitionPlatesConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [VehicleRecognitionPlatesOrderBy!]: [VehicleRecognitionPlatesOrderBy!]`, `condition - VehicleRecognitionPlateCondition: VehicleRecognitionPlateCondition`, `filter - VehicleRecognitionPlateFilter: VehicleRecognitionPlateFilter` |
| `parkingSpaceVehicleSessionVehicleRecognitions` | `ParkingSpaceVehicleSessionVehicleRecognitionsConnection!` |  |
| `first` | `Int` |  Args: `first - Int: Int`, `last - Int: Int`, `offset - Int: Int`, `before - Cursor: Cursor`, `after - Cursor: Cursor`, `orderBy - [ParkingSpaceVehicleSessionVehicleRecognitionsOrderBy!]: [ParkingSpaceVehicleSessionVehicleRecognitionsOrderBy!]`, `condition - ParkingSpaceVehicleSessionVehicleRecognitionCondition: ParkingSpaceVehicleSessionVehicleRecognitionCondition`, `filter - ParkingSpaceVehicleSessionVehicleRecognitionFilter: ParkingSpaceVehicleSessionVehicleRecognitionFilter` |
| `imageUrlSigned` | `String` |  |
| `makeModelImageUrlSigned` | `String` |  |
| `compositeImageUrlSigned` | `String` |  |

## VehicleRecognitionAggregatesFilter (input-object)

Description A filter to be used against aggregates of VehicleRecognition object types.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `sum` | `VehicleRecognitionSumAggregateFilter` |  |
| `distinctCount` | `VehicleRecognitionDistinctCountAggregateFilter` |  |
| `min` | `VehicleRecognitionMinAggregateFilter` |  |
| `max` | `VehicleRecognitionMaxAggregateFilter` |  |
| `average` | `VehicleRecognitionAverageAggregateFilter` |  |
| `stddevSample` | `VehicleRecognitionStddevSampleAggregateFilter` |  |
| `stddevPopulation` | `VehicleRecognitionStddevPopulationAggregateFilter` |  |
| `varianceSample` | `VehicleRecognitionVarianceSampleAggregateFilter` |  |
| `variancePopulation` | `VehicleRecognitionVariancePopulationAggregateFilter` |  |

## VehicleRecognitionAverageAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |
| `dscore` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `cameraId` | `BigFloatFilter` |  |
| `plateBoxXMin` | `BigFloatFilter` |  |
| `plateBoxXMax` | `BigFloatFilter` |  |
| `plateBoxYMin` | `BigFloatFilter` |  |
| `plateBoxYMax` | `BigFloatFilter` |  |
| `imageWidth` | `BigFloatFilter` |  |
| `imageHeight` | `BigFloatFilter` |  |
| `remoteChangeSeqId` | `BigFloatFilter` |  |

## VehicleRecognitionCondition (input-object)

Description A condition to be used against VehicleRecognition object types. All fields are tested for equality and combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigInt` |  |
| `remoteId` | `Int` |  |
| `timestamp` | `Datetime` |  |
| `dscore` | `BigFloat` |  |
| `location` | `String` |  |
| `siteId` | `Int` |  |
| `cameraId` | `Int` |  |
| `remoteImagePath` | `String` |  |
| `imageUrl` | `String` |  |
| `remoteUpdatedAt` | `Datetime` |  |
| `remoteMakeModelImagePath` | `String` |  |
| `makeModelImageUrl` | `String` |  |
| `plateBoxXMin` | `Int` |  |
| `plateBoxXMax` | `Int` |  |
| `plateBoxYMin` | `Int` |  |
| `plateBoxYMax` | `Int` |  |
| `imageWidth` | `Int` |  |
| `imageHeight` | `Int` |  |
| `compositeImageUrl` | `String` |  |
| `remoteChangeSeqId` | `BigInt` |  |
| `updatedAt` | `Datetime` |  |

## VehicleRecognitionDistinctCountAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `remoteId` | `BigIntFilter` |  |
| `timestamp` | `BigIntFilter` |  |
| `dscore` | `BigIntFilter` |  |
| `location` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `cameraId` | `BigIntFilter` |  |
| `remoteImagePath` | `BigIntFilter` |  |
| `imageUrl` | `BigIntFilter` |  |
| `remoteUpdatedAt` | `BigIntFilter` |  |
| `remoteMakeModelImagePath` | `BigIntFilter` |  |
| `makeModelImageUrl` | `BigIntFilter` |  |
| `plateBoxXMin` | `BigIntFilter` |  |
| `plateBoxXMax` | `BigIntFilter` |  |
| `plateBoxYMin` | `BigIntFilter` |  |
| `plateBoxYMax` | `BigIntFilter` |  |
| `imageWidth` | `BigIntFilter` |  |
| `imageHeight` | `BigIntFilter` |  |
| `compositeImageUrl` | `BigIntFilter` |  |
| `remoteChangeSeqId` | `BigIntFilter` |  |
| `updatedAt` | `BigIntFilter` |  |

## VehicleRecognitionFilter (input-object)

Description A filter to be used against VehicleRecognition object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `remoteId` | `IntFilter` |  |
| `timestamp` | `DatetimeFilter` |  |
| `dscore` | `BigFloatFilter` |  |
| `location` | `StringFilter` |  |
| `siteId` | `IntFilter` |  |
| `cameraId` | `IntFilter` |  |
| `remoteImagePath` | `StringFilter` |  |
| `imageUrl` | `StringFilter` |  |
| `remoteUpdatedAt` | `DatetimeFilter` |  |
| `remoteMakeModelImagePath` | `StringFilter` |  |
| `makeModelImageUrl` | `StringFilter` |  |
| `plateBoxXMin` | `IntFilter` |  |
| `plateBoxXMax` | `IntFilter` |  |
| `plateBoxYMin` | `IntFilter` |  |
| `plateBoxYMax` | `IntFilter` |  |
| `imageWidth` | `IntFilter` |  |
| `imageHeight` | `IntFilter` |  |
| `compositeImageUrl` | `StringFilter` |  |
| `remoteChangeSeqId` | `BigIntFilter` |  |
| `updatedAt` | `DatetimeFilter` |  |
| `vehicleRecognitionColorsExist` | `Boolean` |  |
| `vehicleRecognitionMakeModelsExist` | `Boolean` |  |
| `vehicleRecognitionOrientationsExist` | `Boolean` |  |
| `vehicleRecognitionPlates` | `VehicleRecognitionToManyVehicleRecognitionPlateFilter` |  |
| `vehicleRecognitionPlatesExist` | `Boolean` |  |
| `vehicleRecognitionPlateRegionsExist` | `Boolean` |  |
| `ingressVehicleSessionsExist` | `Boolean` |  |
| `egressVehicleSessionsExist` | `Boolean` |  |
| `egressMissedVehicleSessionsExist` | `Boolean` |  |
| `parkingSpaceVehicleSessionVehicleRecognitions` | `VehicleRecognitionToManyParkingSpaceVehicleSessionVehicleRecognitionFilter` |  |
| `parkingSpaceVehicleSessionVehicleRecognitionsExist` | `Boolean` |  |
| `cameraExists` | `Boolean` |  |

## VehicleRecognitionHavingAverageInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `cameraId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `plateBoxXMin` | `HavingIntFilter` |  |
| `plateBoxXMax` | `HavingIntFilter` |  |
| `plateBoxYMin` | `HavingIntFilter` |  |
| `plateBoxYMax` | `HavingIntFilter` |  |
| `imageWidth` | `HavingIntFilter` |  |
| `imageHeight` | `HavingIntFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## VehicleRecognitionHavingDistinctCountInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `cameraId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `plateBoxXMin` | `HavingIntFilter` |  |
| `plateBoxXMax` | `HavingIntFilter` |  |
| `plateBoxYMin` | `HavingIntFilter` |  |
| `plateBoxYMax` | `HavingIntFilter` |  |
| `imageWidth` | `HavingIntFilter` |  |
| `imageHeight` | `HavingIntFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## VehicleRecognitionHavingInput (input-object)

Description Conditions for VehicleRecognition aggregates.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `AND` | `[VehicleRecognitionHavingInput!]` |  |
| `OR` | `[VehicleRecognitionHavingInput!]` |  |
| `sum` | `VehicleRecognitionHavingSumInput` |  |
| `distinctCount` | `VehicleRecognitionHavingDistinctCountInput` |  |
| `min` | `VehicleRecognitionHavingMinInput` |  |
| `max` | `VehicleRecognitionHavingMaxInput` |  |
| `average` | `VehicleRecognitionHavingAverageInput` |  |
| `stddevSample` | `VehicleRecognitionHavingStddevSampleInput` |  |
| `stddevPopulation` | `VehicleRecognitionHavingStddevPopulationInput` |  |
| `varianceSample` | `VehicleRecognitionHavingVarianceSampleInput` |  |
| `variancePopulation` | `VehicleRecognitionHavingVariancePopulationInput` |  |

## VehicleRecognitionHavingMaxInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `cameraId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `plateBoxXMin` | `HavingIntFilter` |  |
| `plateBoxXMax` | `HavingIntFilter` |  |
| `plateBoxYMin` | `HavingIntFilter` |  |
| `plateBoxYMax` | `HavingIntFilter` |  |
| `imageWidth` | `HavingIntFilter` |  |
| `imageHeight` | `HavingIntFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## VehicleRecognitionHavingMinInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `cameraId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `plateBoxXMin` | `HavingIntFilter` |  |
| `plateBoxXMax` | `HavingIntFilter` |  |
| `plateBoxYMin` | `HavingIntFilter` |  |
| `plateBoxYMax` | `HavingIntFilter` |  |
| `imageWidth` | `HavingIntFilter` |  |
| `imageHeight` | `HavingIntFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## VehicleRecognitionHavingStddevPopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `cameraId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `plateBoxXMin` | `HavingIntFilter` |  |
| `plateBoxXMax` | `HavingIntFilter` |  |
| `plateBoxYMin` | `HavingIntFilter` |  |
| `plateBoxYMax` | `HavingIntFilter` |  |
| `imageWidth` | `HavingIntFilter` |  |
| `imageHeight` | `HavingIntFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## VehicleRecognitionHavingStddevSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `cameraId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `plateBoxXMin` | `HavingIntFilter` |  |
| `plateBoxXMax` | `HavingIntFilter` |  |
| `plateBoxYMin` | `HavingIntFilter` |  |
| `plateBoxYMax` | `HavingIntFilter` |  |
| `imageWidth` | `HavingIntFilter` |  |
| `imageHeight` | `HavingIntFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## VehicleRecognitionHavingSumInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `cameraId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `plateBoxXMin` | `HavingIntFilter` |  |
| `plateBoxXMax` | `HavingIntFilter` |  |
| `plateBoxYMin` | `HavingIntFilter` |  |
| `plateBoxYMax` | `HavingIntFilter` |  |
| `imageWidth` | `HavingIntFilter` |  |
| `imageHeight` | `HavingIntFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## VehicleRecognitionHavingVariancePopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `cameraId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `plateBoxXMin` | `HavingIntFilter` |  |
| `plateBoxXMax` | `HavingIntFilter` |  |
| `plateBoxYMin` | `HavingIntFilter` |  |
| `plateBoxYMax` | `HavingIntFilter` |  |
| `imageWidth` | `HavingIntFilter` |  |
| `imageHeight` | `HavingIntFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## VehicleRecognitionHavingVarianceSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `timestamp` | `HavingDatetimeFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `cameraId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `plateBoxXMin` | `HavingIntFilter` |  |
| `plateBoxXMax` | `HavingIntFilter` |  |
| `plateBoxYMin` | `HavingIntFilter` |  |
| `plateBoxYMax` | `HavingIntFilter` |  |
| `imageWidth` | `HavingIntFilter` |  |
| `imageHeight` | `HavingIntFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |
| `updatedAt` | `HavingDatetimeFilter` |  |

## VehicleRecognitionMaxAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `remoteId` | `IntFilter` |  |
| `timestamp` | `DatetimeFilter` |  |
| `dscore` | `BigFloatFilter` |  |
| `siteId` | `IntFilter` |  |
| `cameraId` | `IntFilter` |  |
| `remoteUpdatedAt` | `DatetimeFilter` |  |
| `plateBoxXMin` | `IntFilter` |  |
| `plateBoxXMax` | `IntFilter` |  |
| `plateBoxYMin` | `IntFilter` |  |
| `plateBoxYMax` | `IntFilter` |  |
| `imageWidth` | `IntFilter` |  |
| `imageHeight` | `IntFilter` |  |
| `remoteChangeSeqId` | `BigIntFilter` |  |
| `updatedAt` | `DatetimeFilter` |  |

## VehicleRecognitionMinAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `remoteId` | `IntFilter` |  |
| `timestamp` | `DatetimeFilter` |  |
| `dscore` | `BigFloatFilter` |  |
| `siteId` | `IntFilter` |  |
| `cameraId` | `IntFilter` |  |
| `remoteUpdatedAt` | `DatetimeFilter` |  |
| `plateBoxXMin` | `IntFilter` |  |
| `plateBoxXMax` | `IntFilter` |  |
| `plateBoxYMin` | `IntFilter` |  |
| `plateBoxYMax` | `IntFilter` |  |
| `imageWidth` | `IntFilter` |  |
| `imageHeight` | `IntFilter` |  |
| `remoteChangeSeqId` | `BigIntFilter` |  |
| `updatedAt` | `DatetimeFilter` |  |

## VehicleRecognitionPlate (object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodeId` | `ID!` |  |
| `id` | `BigInt!` |  |
| `remoteId` | `Int!` |  |
| `plate` | `String!` |  |
| `score` | `BigFloat!` |  |
| `primary` | `Boolean!` |  |
| `vehicleRecognitionId` | `BigInt!` |  |
| `siteId` | `Int!` |  |
| `remoteUpdatedAt` | `Datetime!` |  |
| `remoteChangeSeqId` | `BigInt` |  |
| `site` | `Site` |  |
| `vehicleRecognition` | `VehicleRecognition` |  |

## VehicleRecognitionPlateAggregatesFilter (input-object)

Description A filter to be used against aggregates of VehicleRecognitionPlate object types.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `sum` | `VehicleRecognitionPlateSumAggregateFilter` |  |
| `distinctCount` | `VehicleRecognitionPlateDistinctCountAggregateFilter` |  |
| `min` | `VehicleRecognitionPlateMinAggregateFilter` |  |
| `max` | `VehicleRecognitionPlateMaxAggregateFilter` |  |
| `average` | `VehicleRecognitionPlateAverageAggregateFilter` |  |
| `stddevSample` | `VehicleRecognitionPlateStddevSampleAggregateFilter` |  |
| `stddevPopulation` | `VehicleRecognitionPlateStddevPopulationAggregateFilter` |  |
| `varianceSample` | `VehicleRecognitionPlateVarianceSampleAggregateFilter` |  |
| `variancePopulation` | `VehicleRecognitionPlateVariancePopulationAggregateFilter` |  |

## VehicleRecognitionPlateAverageAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |
| `score` | `BigFloatFilter` |  |
| `vehicleRecognitionId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `remoteChangeSeqId` | `BigFloatFilter` |  |

## VehicleRecognitionPlateCondition (input-object)

Description A condition to be used against VehicleRecognitionPlate object types. All fields are tested for equality and combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigInt` |  |
| `remoteId` | `Int` |  |
| `plate` | `String` |  |
| `score` | `BigFloat` |  |
| `primary` | `Boolean` |  |
| `vehicleRecognitionId` | `BigInt` |  |
| `siteId` | `Int` |  |
| `remoteUpdatedAt` | `Datetime` |  |
| `remoteChangeSeqId` | `BigInt` |  |

## VehicleRecognitionPlateDistinctCountAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `remoteId` | `BigIntFilter` |  |
| `plate` | `BigIntFilter` |  |
| `score` | `BigIntFilter` |  |
| `primary` | `BigIntFilter` |  |
| `vehicleRecognitionId` | `BigIntFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `remoteUpdatedAt` | `BigIntFilter` |  |
| `remoteChangeSeqId` | `BigIntFilter` |  |

## VehicleRecognitionPlateFilter (input-object)

Description A filter to be used against VehicleRecognitionPlate object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `remoteId` | `IntFilter` |  |
| `plate` | `StringFilter` |  |
| `score` | `BigFloatFilter` |  |
| `primary` | `BooleanFilter` |  |
| `vehicleRecognitionId` | `BigIntFilter` |  |
| `siteId` | `IntFilter` |  |
| `remoteUpdatedAt` | `DatetimeFilter` |  |
| `remoteChangeSeqId` | `BigIntFilter` |  |
| `vehicleSessionBestAttributesExist` | `Boolean` |  |
| `parkingSpaceVehicleSessionBestAttributesExist` | `Boolean` |  |
| `automationRuleExecutionPlateRecognizedsBySiteIdAndVehicleRecognitionPlateIdExist` | `Boolean` |  |

## VehicleRecognitionPlateHavingAverageInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `vehicleRecognitionId` | `HavingBigintFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |

## VehicleRecognitionPlateHavingDistinctCountInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `vehicleRecognitionId` | `HavingBigintFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |

## VehicleRecognitionPlateHavingInput (input-object)

Description Conditions for VehicleRecognitionPlate aggregates.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `AND` | `[VehicleRecognitionPlateHavingInput!]` |  |
| `OR` | `[VehicleRecognitionPlateHavingInput!]` |  |
| `sum` | `VehicleRecognitionPlateHavingSumInput` |  |
| `distinctCount` | `VehicleRecognitionPlateHavingDistinctCountInput` |  |
| `min` | `VehicleRecognitionPlateHavingMinInput` |  |
| `max` | `VehicleRecognitionPlateHavingMaxInput` |  |
| `average` | `VehicleRecognitionPlateHavingAverageInput` |  |
| `stddevSample` | `VehicleRecognitionPlateHavingStddevSampleInput` |  |
| `stddevPopulation` | `VehicleRecognitionPlateHavingStddevPopulationInput` |  |
| `varianceSample` | `VehicleRecognitionPlateHavingVarianceSampleInput` |  |
| `variancePopulation` | `VehicleRecognitionPlateHavingVariancePopulationInput` |  |

## VehicleRecognitionPlateHavingMaxInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `vehicleRecognitionId` | `HavingBigintFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |

## VehicleRecognitionPlateHavingMinInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `vehicleRecognitionId` | `HavingBigintFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |

## VehicleRecognitionPlateHavingStddevPopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `vehicleRecognitionId` | `HavingBigintFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |

## VehicleRecognitionPlateHavingStddevSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `vehicleRecognitionId` | `HavingBigintFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |

## VehicleRecognitionPlateHavingSumInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `vehicleRecognitionId` | `HavingBigintFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |

## VehicleRecognitionPlateHavingVariancePopulationInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `vehicleRecognitionId` | `HavingBigintFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |

## VehicleRecognitionPlateHavingVarianceSampleInput (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `HavingBigintFilter` |  |
| `remoteId` | `HavingIntFilter` |  |
| `vehicleRecognitionId` | `HavingBigintFilter` |  |
| `siteId` | `HavingIntFilter` |  |
| `remoteUpdatedAt` | `HavingDatetimeFilter` |  |
| `remoteChangeSeqId` | `HavingBigintFilter` |  |

## VehicleRecognitionPlateMaxAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `remoteId` | `IntFilter` |  |
| `score` | `BigFloatFilter` |  |
| `vehicleRecognitionId` | `BigIntFilter` |  |
| `siteId` | `IntFilter` |  |
| `remoteUpdatedAt` | `DatetimeFilter` |  |
| `remoteChangeSeqId` | `BigIntFilter` |  |

## VehicleRecognitionPlateMinAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigIntFilter` |  |
| `remoteId` | `IntFilter` |  |
| `score` | `BigFloatFilter` |  |
| `vehicleRecognitionId` | `BigIntFilter` |  |
| `siteId` | `IntFilter` |  |
| `remoteUpdatedAt` | `DatetimeFilter` |  |
| `remoteChangeSeqId` | `BigIntFilter` |  |

## VehicleRecognitionPlateStddevPopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |
| `score` | `BigFloatFilter` |  |
| `vehicleRecognitionId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `remoteChangeSeqId` | `BigFloatFilter` |  |

## VehicleRecognitionPlateStddevSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |
| `score` | `BigFloatFilter` |  |
| `vehicleRecognitionId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `remoteChangeSeqId` | `BigFloatFilter` |  |

## VehicleRecognitionPlateSumAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `remoteId` | `BigIntFilter` |  |
| `score` | `BigFloatFilter` |  |
| `vehicleRecognitionId` | `BigFloatFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `remoteChangeSeqId` | `BigFloatFilter` |  |

## VehicleRecognitionPlateVariancePopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |
| `score` | `BigFloatFilter` |  |
| `vehicleRecognitionId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `remoteChangeSeqId` | `BigFloatFilter` |  |

## VehicleRecognitionPlateVarianceSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |
| `score` | `BigFloatFilter` |  |
| `vehicleRecognitionId` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `remoteChangeSeqId` | `BigFloatFilter` |  |

## VehicleRecognitionPlatesConnection (object)

Description A connection to a list of VehicleRecognitionPlate values.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodes` | `[VehicleRecognitionPlate!]!` |  |
| `totalCount` | `Int!` |  |

## VehicleRecognitionPlatesOrderBy (enum)

Description Methods to use when ordering VehicleRecognitionPlate .

## VehicleRecognitionStddevPopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |
| `dscore` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `cameraId` | `BigFloatFilter` |  |
| `plateBoxXMin` | `BigFloatFilter` |  |
| `plateBoxXMax` | `BigFloatFilter` |  |
| `plateBoxYMin` | `BigFloatFilter` |  |
| `plateBoxYMax` | `BigFloatFilter` |  |
| `imageWidth` | `BigFloatFilter` |  |
| `imageHeight` | `BigFloatFilter` |  |
| `remoteChangeSeqId` | `BigFloatFilter` |  |

## VehicleRecognitionStddevSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |
| `dscore` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `cameraId` | `BigFloatFilter` |  |
| `plateBoxXMin` | `BigFloatFilter` |  |
| `plateBoxXMax` | `BigFloatFilter` |  |
| `plateBoxYMin` | `BigFloatFilter` |  |
| `plateBoxYMax` | `BigFloatFilter` |  |
| `imageWidth` | `BigFloatFilter` |  |
| `imageHeight` | `BigFloatFilter` |  |
| `remoteChangeSeqId` | `BigFloatFilter` |  |

## VehicleRecognitionSumAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `remoteId` | `BigIntFilter` |  |
| `dscore` | `BigFloatFilter` |  |
| `siteId` | `BigIntFilter` |  |
| `cameraId` | `BigIntFilter` |  |
| `plateBoxXMin` | `BigIntFilter` |  |
| `plateBoxXMax` | `BigIntFilter` |  |
| `plateBoxYMin` | `BigIntFilter` |  |
| `plateBoxYMax` | `BigIntFilter` |  |
| `imageWidth` | `BigIntFilter` |  |
| `imageHeight` | `BigIntFilter` |  |
| `remoteChangeSeqId` | `BigFloatFilter` |  |

## VehicleRecognitionToManyParkingSpaceVehicleSessionVehicleRecognitionFilter (input-object)

Description A filter to be used against many ParkingSpaceVehicleSessionVehicleRecognition object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `ParkingSpaceVehicleSessionVehicleRecognitionAggregatesFilter` |  |

## VehicleRecognitionToManyVehicleRecognitionPlateFilter (input-object)

Description A filter to be used against many VehicleRecognitionPlate object types. All fields are combined with a logical ‘and.’

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `aggregates` | `VehicleRecognitionPlateAggregatesFilter` |  |

## VehicleRecognitionVariancePopulationAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |
| `dscore` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `cameraId` | `BigFloatFilter` |  |
| `plateBoxXMin` | `BigFloatFilter` |  |
| `plateBoxXMax` | `BigFloatFilter` |  |
| `plateBoxYMin` | `BigFloatFilter` |  |
| `plateBoxYMax` | `BigFloatFilter` |  |
| `imageWidth` | `BigFloatFilter` |  |
| `imageHeight` | `BigFloatFilter` |  |
| `remoteChangeSeqId` | `BigFloatFilter` |  |

## VehicleRecognitionVarianceSampleAggregateFilter (input-object)

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `BigFloatFilter` |  |
| `remoteId` | `BigFloatFilter` |  |
| `dscore` | `BigFloatFilter` |  |
| `siteId` | `BigFloatFilter` |  |
| `cameraId` | `BigFloatFilter` |  |
| `plateBoxXMin` | `BigFloatFilter` |  |
| `plateBoxXMax` | `BigFloatFilter` |  |
| `plateBoxYMin` | `BigFloatFilter` |  |
| `plateBoxYMax` | `BigFloatFilter` |  |
| `imageWidth` | `BigFloatFilter` |  |
| `imageHeight` | `BigFloatFilter` |  |
| `remoteChangeSeqId` | `BigFloatFilter` |  |

## VehicleRecognitionsConnection (object)

Description A connection to a list of VehicleRecognition values.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `nodes` | `[VehicleRecognition!]!` |  |
| `totalCount` | `Int!` |  |

## VehicleRecognitionsOrderBy (enum)

Description Methods to use when ordering VehicleRecognition .
