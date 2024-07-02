import { createSlice } from "@reduxjs/toolkit";

export enum ActionSecurityRisk {
  UNKNOWN = -1,
  LOW = 0,
  MEDIUM = 1,
  HIGH = 2
}

export type Invariant = {
  content: string;
  security_risk: ActionSecurityRisk;
};

const initialLogs: Invariant[] = [];

export const invariantSlice = createSlice({
  name: "invariant",
  initialState: {
    logs: initialLogs,
  },
  reducers: {
    appendInvariantInput: (state, action) => {
      state.logs.push({ content: action.payload.command || action.payload.code || action.payload.content , security_risk: action.payload.security_risk as ActionSecurityRisk });
    },
  },
});

export const { appendInvariantInput } = invariantSlice.actions;

export default invariantSlice.reducer;